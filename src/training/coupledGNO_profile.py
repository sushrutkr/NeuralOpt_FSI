import os
import sys
import random
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import pytorch_memlab
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import HeteroData
import torch.profiler
from pytorch_memlab import MemReporter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from model.neuralFSI import *
from dataloader.dataload import *

from torch.fx import symbolic_trace, GraphModule
import torch.profiler

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
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
    
    params_network = {
        'memb_net': {
            'inNodeFeatures': 10,
            'nNodeFeatEmbedding': 24,
            'outNodeFeatures': 10,
            'nEdgeFeatures': 7,
            'ker_width': 8
        },
        'flow_net': {
            'inNodeFeatures': 4,
            'nNodeFeatEmbedding': 24,
            'outNodeFeatures': 4,
            'nEdgeFeatures': 14,
            'ker_width': 4
        },
        'attn_dim': 24,
        'nlayers': 10,  # Keep at 4 for now
        'time_embedding_dim': 16
    }
    
    params_training = {
        'epochs': 3,
        'learning_rate': 0.001,
        'scheduler_step': 500,
        'scheduler_gamma': 0.5,
        'validation_frequency': 100,
        'save_frequency': 100,
    }

    params_data = {
        'batch_size': 3,
        'ntsteps': 1,
        'val_split': 0.3,
        'reload_data': True,
        'cache_loc': "/home/skumar94/scr16_rmittal3/skumar94/GNO_FSI/TrainingData/data_train_cache.pt"
    }

    train_radius = {
        'radius_flow': 0.08,
        'radius_memb': 0.04,
        'radius_cross': 0.04
    }

    train_loader, val_loader = dataloader(train_radius, 
                                         params_data['batch_size'],
                                         params_data['ntsteps'],
                                         params_data['val_split'],
                                         loadData = params_data["reload_data"],
										 cache_file=params_data["cache_loc"])
    
    print("----Loaded Data----")

    model_instance = neuralFSI(params=params_network).to(device)

    reporter = MemReporter(model_instance)

    scaler = torch.cuda.amp.GradScaler()

    total_params = sum(p.numel() for p in model_instance.parameters())
    print(f"Number of parameters: {total_params}")
    param_mem = sum(p.element_size() * p.nelement() 
                    for p in model_instance.parameters())
    print(f"True parameter memory: {param_mem/1e6}MB")

    optimizer = torch.optim.Adam(model_instance.parameters(), 
                                lr=params_training['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=params_training['scheduler_step'], 
                                               gamma=params_training['scheduler_gamma'])

    criterion = torch.nn.MSELoss()

    start_epoch = 0
    best_val_loss = float('inf')

    profiler_log_dir = './logs/densenet_profile_detailed'
    os.makedirs(profiler_log_dir, exist_ok=True)

    # Training loop with DenseNet profiling
    for epoch in range(start_epoch, params_training['epochs']):
        model_instance.train()
        train_loss = 0.0
        flow_loss_batch = 0.0
        memb_loss_batch = 0.0
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        ) as prof:
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                batch = batch.to(device)
                
                torch.cuda.reset_peak_memory_stats(device)
                start_mem = torch.cuda.memory_allocated(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out_flow, out_memb = model_instance(batch)
                    loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
                    loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))
                    loss = loss_flow + loss_memb
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                print(f"\n--- Memory Report after batch {batch_idx} ---")
                reporter.report()
                
                peak_mem = torch.cuda.max_memory_allocated(device)
                delta_mem = torch.cuda.memory_allocated(device) - start_mem
                
                print(f"Batch {batch_idx} | "
                      f"Peak: {peak_mem/1e9:.2f}GB | "
                      f"Delta: {delta_mem/1e9:.2f}GB | "
                      f"Cache: {torch.cuda.memory_reserved(device)/1e9:.2f}GB")
                
                train_loss += loss.item()
                flow_loss_batch += loss_flow.item()
                memb_loss_batch += loss_memb.item()
                
                prof.step()

        print(f"\n=== Epoch {epoch} DenseNet Memory Profile ===")
        print(prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_memory_usage",
            row_limit=20,
            max_name_column_width=60,
            max_shapes_column_width=80
        ))
        
        print(f"\nEpoch {epoch} Memory Summary:")
        print(f"Max allocated: {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB")
        print(f"Max reserved: {torch.cuda.max_memory_reserved(device)/1e9:.2f}GB")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{params_training['epochs']}, "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Flow loss: {flow_loss_batch/len(train_loader):.6f}, "
              f"Memb loss: {memb_loss_batch/len(train_loader):.6f}, "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

if __name__ == "__main__":
    main()