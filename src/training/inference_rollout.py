import os
import sys
import random
from timeit import default_timer
import numpy as np
import torch
import json
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import ks_2samp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from model.neuralFSI import *
from dataloader.dataload import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(checkpoint_path='./utils/best_model.pth', start_timeframe=0, num_steps=100, dt=0.01, output_dir='./inference_results'):
    
    set_seed(42)

    with open("./input/config.json", "r") as f:
        config = json.load(f)

    params_network = config["params_network"]
    params_training = config["params_training"]
    params_data = config["params_data"]
    train_radius = config["train_radius"]

    print("Network Parameters:", params_network)
    print("Training Parameters:", params_training)
    print("Data Parameters:", params_data)
    print("Train Radius:", train_radius)

    _, val_loader = dataGenerate.data_loader_multiGPU(
        train_radius, 
        params_data['batch_size'],
        params_data['ntsteps'],
        params_data['val_split'],
        world_size=1,
        rank=0,
        loadData = params_data["reload_data"],
        cache_loc=params_data["cache_loc"]
    )
    
    print("Data loaded")

    model_instance = neuralFSI(params=params_network).to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)        
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        raise ValueError("Checkpoint path required for inference")

    model_instance.eval()

    # Prepare initial graph from a timeframe in validation set
    initial_batch = next(iter(val_loader))
    initial_graph = initial_batch[0] if isinstance(initial_batch, list) else initial_batch  # Take first graph if batched
    initial_graph = initial_graph.to(device)
    
    current_time = torch.tensor([start_timeframe * dt], dtype=torch.float32, device=device)
    
    # Autoregressive inference loop
    states = [] 
    current_graph = initial_graph.clone()
    
    with torch.no_grad():
        for step in range(num_steps):
            print(f"Inference step {step+1}/{num_steps}, current time: {current_time.item():.4f}")
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out_flow, out_memb = model_instance(current_graph)
            
            current_graph['flow'].x = out_flow 
            current_graph['memb'].x = out_memb
            
            current_time += dt
            
            states.append({
                'step': step + 1,
                'time': current_time.item(),
                'flow': out_flow.cpu(),
                'memb': out_memb.cpu()
            })
            
            torch.cuda.empty_cache()
    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(states, os.path.join(output_dir, 'inference_states.pt'))
    print(f"Inference completed. States saved to {output_dir}/inference_states.pt")

if __name__ == "__main__":
    # python inference.py --checkpoint ./utils/best_model.pth --start_timeframe 0 --num_steps 100 --dt 0.01
    import argparse
    parser = argparse.ArgumentParser(description="Autoregressive Inference Script")
    parser.add_argument('--checkpoint', type=str, default='./utils/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--start_timeframe', type=int, default=0, help='Starting timeframe index')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of autoregressive steps')
    parser.add_argument('--dt', type=float, default=0.01, help='Timestep delta')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    args = parser.parse_args()
    
    main(checkpoint_path=args.checkpoint, start_timeframe=args.start_timeframe, num_steps=args.num_steps, dt=args.dt, output_dir=args.output_dir)