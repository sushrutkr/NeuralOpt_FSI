import os
import sys
import random
from timeit import default_timer
import numpy as np
import torch
import json
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import ks_2samp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from model.neuralFSI import *
from dataloader.dataload import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reduce_tensor(tensor, world_size):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= world_size
	return rt

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def check_distribution_shift(rank, train_loader, val_loader):
    train_features, val_features = [], []
    for batch in train_loader:
        train_features.append(batch['flow'].x.detach().cpu().numpy().flatten())
        train_features.append(batch['memb'].x.detach().cpu().numpy().flatten())
    for batch in val_loader:
        val_features.append(batch['flow'].x.detach().cpu().numpy().flatten())
        val_features.append(batch['memb'].x.detach().cpu().numpy().flatten())
    train_features = np.concatenate(train_features)
    val_features = np.concatenate(val_features)
    stat, p_value = ks_2samp(train_features, val_features)
    print(f"[{rank}] KS Statistic: {stat:.4f}, P-value: {p_value:.4f}")
    return p_value < 0.05  # Shift if p < 0.05

def main(world_size: int, rank: int, local_rank: int, checkpoint_path=None): 
	torch.cuda.set_device(local_rank)
	dist.init_process_group('nccl', world_size=world_size, rank=rank)
	set_seed(42)

	with open("./input/config.json", "r") as f:
			config = json.load(f)

	params_network = config["params_network"]
	params_training = config["params_training"]
	params_data = config["params_data"]
	train_radius = config["train_radius"]

	if rank == 0:
		print("Network Parameters:", params_network)
		print("Training Parameters:", params_training)
		print("Data Parameters:", params_data)
		print("Train Radius:", train_radius)

	# Load data
	train_loader, val_loader = dataGenerate.data_loader_multiGPU(
		train_radius, 
		params_data['batch_size'],
		params_data['ntsteps'],
		params_data['val_split'],
		world_size,
		rank,
		loadData = params_data["reload_data"],
		cache_loc=params_data["cache_loc"])
	
	train_sampler = train_loader.sampler
	print(f"[rank {rank}] data loaded")

	dist.barrier()

	p_value_shift = check_distribution_shift(rank, train_loader, val_loader)
	print(f"[rank {rank}] Distribution shift detected: {p_value_shift}")

	device = torch.device(f'cuda:{local_rank}')

	model_instance = neuralFSI(params=params_network).to(device)
	model_instance = DDP(model_instance, device_ids=[local_rank], find_unused_parameters=True)

	optimizer = torch.optim.AdamW(model_instance.parameters(),
																lr=params_training['learning_rate'], 
																weight_decay=1e-4)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min', factor=0.5, patience=10, verbose=(rank == 0)
	)

	criterion = torch.nn.MSELoss()

	# Initialize training
	start_epoch = 0
	best_val_loss = float('inf')
	epochs_no_improve = 0
	memb_frozen = False
	patience = 20  # For early stopping

	if checkpoint_path:
		# Load with map_location to handle device differences
		checkpoint = torch.load(checkpoint_path, map_location=device)        
		model_instance.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])        
		start_epoch = checkpoint['epoch'] + 1
		best_val_loss = checkpoint['val_loss']
		epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
		if rank == 0:
			print(f"Resuming training from epoch {start_epoch}")

	#training
	for epoch in range(start_epoch, params_training['epochs']):
		train_sampler.set_epoch(epoch)
		model_instance.train()
		train_loss = 0.0
		flow_loss_batch = 0.0
		memb_loss_batch = 0.0 

		for batch in train_loader:
			optimizer.zero_grad(set_to_none=True)
			batch = batch.to(device)

			with torch.autocast(device_type='cuda', dtype=torch.float16): 
				out_flow, out_memb = model_instance(batch)

				loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
				loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))

			loss = params_training['flow_weight']*loss_flow + params_training['memb_weight']*loss_memb
				
			if torch.isnan(loss):
				raise ValueError("NaN loss detected")
			
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)

			optimizer.step()
			
			train_loss += loss.item()
			flow_loss_batch += loss_flow.item()
			memb_loss_batch += loss_memb.item()

			torch.cuda.empty_cache()
			del out_memb, out_flow, loss_flow, loss_memb
		
		avg_train_loss = torch.tensor((train_loss / len(train_loader)), device=device)
		avg_flow_loss = torch.tensor((flow_loss_batch / len(train_loader)),  device=device)
		avg_memb_loss = torch.tensor((memb_loss_batch / len(train_loader)),  device=device)

		dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
		dist.all_reduce(avg_flow_loss, op=dist.ReduceOp.SUM)
		dist.all_reduce(avg_memb_loss, op=dist.ReduceOp.SUM)

		reduced_loss = avg_train_loss / world_size
		reduced_loss_flow = avg_flow_loss / world_size
		reduced_loss_memb = avg_memb_loss / world_size
		
		if reduced_loss_memb < params_training["freeze_threshold"] and not memb_frozen:
			if rank == 0:
				print(f"Freezing membrane head at epoch {epoch+1}, avg memb loss {reduced_loss_memb:.4e}")
			for p in model_instance.module.encoder["memb"].parameters():
				p.requires_grad = False
			for p in model_instance.module.decoder["memb"].parameters():
				p.requires_grad = False
			memb_frozen = True
		
		if rank == 0:
			print(
				f"Epoch {epoch+1}/{params_training['epochs']}, Train Loss: {reduced_loss:.6f}, "
				f"Flow loss: {reduced_loss_flow:.6f}, "
				f"Memb loss: {reduced_loss_memb:.6f}, "
				f"lr: {optimizer.param_groups[0]['lr']:.6f}"
			)

			# Save model 
			if (epoch + 1) % params_training['save_frequency'] == 0:
				torch.save({
					'epoch': epoch,
					'model_state_dict': model_instance.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
					'train_loss': reduced_loss.item(),
					'val_loss': best_val_loss,
					'epochs_no_improve': epochs_no_improve  # Save for resume
				}, f'./utils/model_epoch_{epoch+1}.pth')
				print(f"Model saved at epoch {epoch+1}")

		# Validation
		do_early_stop = False
		if (epoch + 1) % params_training['validation_frequency'] == 0:
				avg_val_loss_local = 0.0
				epochs_no_improve_local = epochs_no_improve  # Start with current
				if rank == 0:
					model_instance.eval()
					val_loss = 0.0
					with torch.no_grad():
						for batch in val_loader:
							batch = batch.to(device)
							with torch.autocast(device_type='cuda', dtype=torch.float16):
								out_flow, out_memb = model_instance(batch)
								loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
								loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))
								loss = params_training['flow_weight']*loss_flow + params_training['memb_weight']*loss_memb
							val_loss += loss.item()
							del out_memb, out_flow, loss_flow, loss_memb

					avg_val_loss_local = val_loss / len(val_loader)
					print(f"Epoch {epoch+1}/{params_training['epochs']}, Validation Loss: {avg_val_loss_local:.6f}")

					if avg_val_loss_local < best_val_loss:
						best_val_loss = avg_val_loss_local
						torch.save(model_instance.state_dict(), './utils/best_model.pth')
						print(f"Best model saved with validation loss: {best_val_loss:.6f}")
						epochs_no_improve_local = 0
					else:
						epochs_no_improve_local += 1

				# Broadcast to all ranks (all call this)
				broadcast_list = torch.tensor([avg_val_loss_local, epochs_no_improve_local], dtype=torch.float32, device=device)
				dist.broadcast(broadcast_list, src=0)
				avg_val_loss = broadcast_list[0].item()
				epochs_no_improve = int(broadcast_list[1].item())
				
				dist.barrier()
				
				# Step scheduler on all ranks
				scheduler.step(avg_val_loss)

				# Early stopping check (after broadcast, so all ranks agree)
				if epochs_no_improve >= patience:
					do_early_stop = True

		if do_early_stop:
			if rank == 0:
				print(f"Early stopping triggered after {patience} epochs without improvement")
			break

	dist.destroy_process_group()

if __name__ == "__main__":
	checkpoint = None  #'model_epoch_1000.pth'
	world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
	rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
	local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
	if rank == 0:
			print("GPUs available to use : ", world_size)
	main(world_size, rank, local_rank, checkpoint)