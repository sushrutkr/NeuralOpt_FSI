#Run - python egno_inference.py --config_by_file

import argparse
from argparse import Namespace
import torch
import torch.utils.data
from membrane.membraneDataloader import MembraneDataset
from model.egno import EGNO
import os
from torch import nn, optim
import json
import random
import numpy as np

parser = argparse.ArgumentParser(description='EGNO')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N', help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='exp_results', metavar='N', help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N', help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N', help='hidden dim')
parser.add_argument('--model', type=str, default='egno', metavar='N')
parser.add_argument('--n_layers', type=int, default=4, metavar='N', help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N', help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N', help='timing experiment')
parser.add_argument('--delta_frame', type=int, default=30, help='Number of frames delta.')
parser.add_argument('--data_dir', type=str, default='', help='Data directory.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument("--config_by_file", default=None, nargs="?", const='', type=str)
parser.add_argument('--lambda_link', type=float, default=1, help='The weight of the linkage loss.')
parser.add_argument('--n_cluster', type=int, default=3, help='The number of clusters.')
parser.add_argument('--flat', action='store_true', default=False, help='flat MLP')
parser.add_argument('--interaction_layer', type=int, default=3, help='The number of interaction layers per block.')
parser.add_argument('--pooling_layer', type=int, default=3, help='The number of pooling layers in EGPN.')
parser.add_argument('--decoder_layer', type=int, default=1, help='The number of decoder layers.')
parser.add_argument('--case', type=str, default='walk', help='The case, walk or run.')
parser.add_argument('--num_timesteps', type=int, default=1, help='The number of time steps.')
parser.add_argument('--time_emb_dim', type=int, default=32, help='The dimension of time embedding.')
parser.add_argument('--num_modes', type=int, default=2, help='The number of modes.')
parser.add_argument('--model_path', type=str, default= "exp_results/membrane/saved_model.pth", help='Saved model path')

args = parser.parse_args()

if args.config_by_file is not None:
    if len(args.config_by_file) == 0:
        job_param_path = 'configs/config_membrane.json'
    else:
        job_param_path = args.config_by_file
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
    # Only update existing keys
    args = vars(args)
    args.update((k, v) for k, v in hyper_params.items() if k in args)
    args = Namespace(**args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss(reduction='none')

print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

def main():
    # Fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_train = MembraneDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                    delta_frame=args.delta_frame, num_timesteps=args.num_timesteps)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    print(f"Training DataLoader created with {len(loader_train)} batches.")

    if args.model == 'egno':
        model = EGNO(n_layers=args.n_layers, in_node_nf=2, in_edge_nf=2, hidden_nf=args.nf, device=device, with_v=True,
                     flat=args.flat, activation=nn.SiLU(), use_time_conv=True, num_modes=args.num_modes,
                     num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim)
    else:
        raise NotImplementedError('Unknown model:', args.model)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    predictions = predict(model, loader_train)

    np.save("predictions.npy",predictions)

    return

def predict(model, loader):
    model.eval()

    predictions = []
    for batch_idx, data in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}")
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data] #send data to device

        for i in [-1, -2]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        print("Shape of loc:", loc.shape)
        print("Shape of vel:", vel.shape)
        print("Shape of edges:", [e.shape for e in edges])
        print("Shape of edge_attr:", [ea.shape for ea in edge_attr])
        print("Shape of local_edges:", [le.shape for le in local_edges])
        print("Shape of local_edge_fea:", [lef.shape for lef in local_edge_fea])
        print("Shape of Z:", Z.shape)
        print("Shape of loc_end:", loc_end.shape)
        print("Shape of vel_end:", vel_end.shape)

        loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))  # [BN, 3]
        loc = loc.view(-1, loc.size(2))
        vel = vel.view(-1, vel.size(2))
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        edges = torch.cat(list(edges + offset), dim=-1)  # [2, BM]
        edge_attr = torch.cat(list(edge_attr), dim=0)  # [BM, ]
        local_edge_index = torch.cat(list(local_edges + offset), dim=-1)  # [2, BM]
        local_edge_fea = torch.cat(list(local_edge_fea), dim=0)  # [BM, ]
        Z = Z.view(-1, Z.size(2))

        if args.model == 'egno':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred, vel_pred, _ = model(loc, nodes, edges, edge_attr, v=vel, loc_mean=loc_mean)
            
        else:
            raise Exception("Wrong model")
        
        #add reshape loc_pred to loc_end original shape
        loc_pred = loc_pred.view(args.num_timesteps, -1, 3)
        loc_pred = loc_pred.transpose(0, 1)
        loc_pred = loc_pred.view(batch_size, n_nodes, args.num_timesteps, 3)

        vel_pred = vel_pred.view(args.num_timesteps, -1, 3)
        vel_pred = vel_pred.transpose(0, 1)
        vel_pred = vel_pred.view(batch_size, n_nodes, args.num_timesteps, 3)
                
        predictions.append((loc_pred.detach().cpu().numpy(),vel_pred.detach().cpu().numpy()))

    return predictions

if __name__ == "__main__":
    main()