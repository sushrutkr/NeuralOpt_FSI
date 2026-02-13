import os
import sys
import random
from timeit import default_timer
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import HeteroData

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from model.neuralFSI import *
from dataloader.dataload import *

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def main(checkpoint_path=None):
	set_seed(42)

	# Parameters
	params_network = {
		'memb_net': {
			'inNodeFeatures'				: 10, #Current keeping same as out dimension because we want to obtain latent dimension of output
			'nNodeFeatEmbedding'		: 24,
			'outNodeFeatures'				: 10,
			'nEdgeFeatures'					: 7,
			'ker_width'							: 8
		},
		'flow_net': {
			'inNodeFeatures'				: 4,
			'nNodeFeatEmbedding'		: 24,
			'outNodeFeatures'				: 4,
			'nEdgeFeatures'					: 14,
			'ker_width'							: 4
		},
		'attn_dim'								: 24, #found 16 to be a better value compared to 8, 24, 32,
		'nlayers'									: 4,
		'time_embedding_dim'			: 8
	}
	
	params_training = {
		'epochs' 								: 10,
		'learning_rate' 				: 0.001 ,
		'scheduler_step' 				: 500,  
		'scheduler_gamma' 			: 0.5,
		'validation_frequency' 	: 100,
		'save_frequency' 				: 100,
	}

	params_data = {
		'batch_size' 		: 3,
		'ntsteps' 			: 2, #extra from current one
		'val_split'			: 0.3
	}

	train_radius = {
		'radius_flow' 	: 0.08,		 # keeping it >=
		'radius_memb'		: 0.04,		 #makes sense to keep <= 2X\Delta_memb
		'radius_cross'  : 0.04     #makes sense to keep <= 2X\Delta_memb
	}

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load data
	train_loader, val_loader = dataloader(train_radius, 
																				params_data['batch_size'],
																				params_data['ntsteps'],
																				params_data['val_split'],
																				loadData = True,
																				cache_file="/home/skumar94/scr16_rmittal3/skumar94/GNO_FSI/TrainingData/data_train_cache.pt")
	

	# train_loader, val_loader, scaler = dataloader_test('../sample_data/', train_radius['radius_flow'], 
	# 																									params_data['batch_size'], 
	# 																									timeRange['start'],
	# 																									timeRange['end'],
	# 																									timeRange['gap'], params_data['ntsteps'])
	
	print("----Loaded Data----")

	# Initialize model
	# For now only performing next time step prediction
	model_instance = neuralFSI(params=params_network).to(device)

	total_params = sum(p.numel() for p in model_instance.parameters())
	print(f"Number of parameters: {total_params}")

	print(model_instance)

	optimizer = torch.optim.Adam(model_instance.parameters(), 
															lr=params_training['learning_rate'])

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
																						 step_size=params_training['scheduler_step'], 
																						 gamma=params_training['scheduler_gamma'])

	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
	criterion = torch.nn.MSELoss()

	# model_instance = torch.compile(model_instance, dynamic=True)

	# Initialize training
	start_epoch = 0
	if checkpoint_path:
		checkpoint = torch.load(checkpoint_path)        
		model_instance.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])        
		start_epoch = checkpoint['epoch'] + 1
		best_val_loss = checkpoint['val_loss']
		print(f"Resuming training from epoch {start_epoch}")
	else:
		best_val_loss = float('inf')

	freeze_threshold = 1e-3
	memb_frozen = False

	#training
	for epoch in range(start_epoch, params_training['epochs']):
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
			loss = loss_flow + loss_memb
				
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)

			optimizer.step()

			#Freezing membrane model after reaching a the
			if (loss_memb.item()<freeze_threshold) and (not memb_frozen):
				print(f"Freezing membrane head at epoch {epoch}, batch loss {loss_memb:.4e}")
				# for p in model_instance.module.encoder["memb"].parameters(): !use module when using DDP
				for p in model_instance.encoder["memb"].parameters(): 
					p.requires_grad = False
				for p in model_instance.decoder["memb"].parameters():
					p.requires_grad = False
				memb_frozen = True
							

			train_loss += loss.item()
			flow_loss_batch += loss_flow.item()
			memb_loss_batch += loss_memb.item()
			torch.cuda.empty_cache()
			del out_memb, out_flow, loss_flow, loss_memb
		
		avg_train_loss = train_loss / len(train_loader)
		print(
				f"Epoch {epoch+1}/{params_training['epochs']}, Train Loss: {avg_train_loss:.6f}, "
				f"Flow loss: {flow_loss_batch/len(train_loader):.6f}, "
				f"Memb loss: {memb_loss_batch/len(train_loader):.6f}, "
				f"lr: {optimizer.param_groups[0]['lr']:.6f}"
		)

		# Save model 
		if (epoch + 1) % params_training['save_frequency'] == 0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model_instance.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'train_loss': avg_train_loss,
				'val_loss': best_val_loss
			}, f'./utils/model_epoch_{epoch+1}.pth')
			print(f"Model saved at epoch {epoch+1}")

		# Validation
		if (epoch + 1) % params_training['validation_frequency'] == 0:
			model_instance.eval()
			val_loss = 0.0
			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(device)
					with torch.autocast(device_type='cuda', dtype=torch.float16):
						out_flow, out_memb = model_instance(batch)

						loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
						loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))
						loss = loss_flow + loss_memb

					del out_memb, out_flow, loss_flow, loss_memb
					val_loss += loss.item()

			avg_val_loss = val_loss / len(val_loader)
			print(f"Epoch {epoch+1}/{params_training['epochs']}, Validation Loss: {avg_val_loss:.6f}")

			# Save best model
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				torch.save(model_instance.state_dict(), './utils/best_model.pth')
				print(f"Best model saved with validation loss: {best_val_loss:.6f}")

		scheduler.step()


if __name__ == "__main__":
	main()
	# main('./utils/model_epoch_100.pth')