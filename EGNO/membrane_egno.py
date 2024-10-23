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
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

# Argument parsing
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
parser.add_argument('--data_dir', type=str, default='membrane/1e6', help='Data directory.')
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
args = parser.parse_args()

if args.config_by_file is not None:
    if len(args.config_by_file) == 0:
        job_param_path = 'configs/config_membrane.json'
    else:
        job_param_path = args.config_by_file
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
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

    dataset_train = MembraneDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir, delta_frame=args.delta_frame, num_timesteps=args.num_timesteps)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    print(f"Training DataLoader created with {len(loader_train)} batches.")

    if args.model == 'egno':
        model = EGNO(n_layers=args.n_layers, in_node_nf=2, in_edge_nf=2, hidden_nf=args.nf, device=device, with_v=True, flat=args.flat, activation=nn.SiLU(), use_time_conv=True, num_modes=args.num_modes, num_timesteps=args.num_timesteps, time_emb_dim=args.time_emb_dim)
    else:
        raise NotImplementedError('Unknown model:', args.model)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_save_path = os.path.join(args.outf, args.exp_name, 'saved_model.pth')
    print(f'Model saved to {model_save_path}')

    results = {'eval epoch': [], 'val loss': [], 'test loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    best_lp_loss = 1e8

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(vars(args))

        for epoch in range(0, args.epochs):
            train_loss, lp_loss = train(model, optimizer, epoch, loader_train)
            results['train loss'].append(train_loss)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lp_loss", lp_loss, step=epoch)

            # Log learning rate
            for param_group in optimizer.param_groups:
                mlflow.log_metric("learning_rate", param_group['lr'], step=epoch)

        # Save the model
        torch.save(model.state_dict(), model_save_path)
        mlflow.pytorch.log_model(model, "model")

        # Save the results to a JSON file
        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)

        print(f'Model saved to {model_save_path} at epoch {epoch + 1}')

    return best_train_loss, best_val_loss, best_test_loss, best_epoch, best_lp_loss

def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'lp_loss': 0}

    for batch_idx, data in enumerate(loader):
        print(f"Processing batch {batch_idx + 1}/{len(loader)}")
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]

        for i in [-1, -2]:
            d = data[i].view(batch_size * n_nodes, args.num_timesteps, 3)
            data[i] = d.transpose(0, 1).contiguous().view(-1, 3)

        loc, vel, edges, edge_attr, local_edges, local_edge_fea, Z, loc_end, vel_end = data
        # print("Shape of loc:", loc.shape)  #Initial Loc
        # print("Shape of vel:", vel.shape)
        # print("Shape of edges:", [e.shape for e in edges])
        # print("Shape of edge_attr:", [ea.shape for ea in edge_attr])
        # print("Shape of local_edges:", [le.shape for le in local_edges])
        # print("Shape of local_edge_fea:", [lef.shape for lef in local_edge_fea])
        # print("Shape of Z:", Z.shape)
        # print("Shape of loc_end:", loc_end.shape) #Location of all points in time
        # print("Shape of vel_end:", vel_end.shape)

        loc_mean = loc.mean(dim=1, keepdim=True).repeat(1, n_nodes, 1).view(-1, loc.size(2))  # [BN, 3]
        loc = loc.view(-1, loc.size(2))
        vel = vel.view(-1, vel.size(2))
        offset = (torch.arange(batch_size) * n_nodes).unsqueeze(-1).unsqueeze(-1).to(edges.device)
        #This line creates a tensor of shape [batch_size, 1, 1] that contains the values [0, n_nodes, 2*n_nodes, ..., (batch_size-1)*n_nodes]. 
        
        edges = torch.cat(list(edges + offset), dim=-1)  # [2, BM]
        
        edge_attr = torch.cat(list(edge_attr), dim=0)  # [BM, ]
        local_edge_index = torch.cat(list(local_edges + offset), dim=-1)  # [2, BM]
        local_edge_fea = torch.cat(list(local_edge_fea), dim=0)  # [BM, ]
        Z = Z.view(-1, Z.size(2))

        optimizer.zero_grad()

        if args.model == 'egno':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred, vel_pred, _ = model(loc, nodes, edges, edge_attr, v=vel, loc_mean=loc_mean)
        else:
            raise Exception("Wrong model")

        losses = loss_mse(loc_pred, loc_end).view(args.num_timesteps, batch_size * n_nodes, 3)
        losses = torch.mean(losses, dim=(1, 2))
        loss = torch.mean(losses)

        if backprop:
            loss.backward()
            optimizer.step()

        res['loss'] += losses[-1].item() * batch_size
        res['counter'] += batch_size

    if res['counter'] == 0:
        raise ValueError("No batches were processed. Please check your data loader and dataset.")

    prefix = "==> " if not backprop else ""
    print('%s epoch %d avg loss: %.5f avg lploss: %.5f lr: %.5f' % (
        prefix + loader.dataset.partition, epoch, res['loss'] / res['counter'], res['lp_loss'] / res['counter'], optimizer.param_groups[0]['lr']
    ))

    return res['loss'] / res['counter'], res['lp_loss'] / res['counter']

if __name__ == "__main__":
    main()