import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import os
import re

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        print(self.file_path)
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            std = self.std[sample_idx]+ self.eps # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class generateDataset:
    def __init__(self, ninit, nend, ngap, splitLen=1, folder="./"):
        self.nodes = []
        self.elem = []
        self.ninit = ninit
        self.nend = nend 
        self.ngap = ngap
        self.split = splitLen
        self.ntsteps = int(((nend - ninit) / ngap) + 1)
        self.folder = folder
        fnameMesh = os.path.join(folder, "marker.{:>07d}.dat".format(ninit))
        self.nNodes, self.nElem = generateDataset.obtainNnodesAndElem(fnameMesh)
        self.AllNodes = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
        self.AllVel = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
        self.AllElem = np.zeros(shape=(self.nElem, 3, self.ntsteps))
        self.SplitNodes = np.array([])
        self.SplitVel = np.array([])
        self.SplitElem = np.array([])
        self.compileData()
        self.splitData()

    @staticmethod
    def obtainNnodesAndElem(fnameMesh):
        with open(fnameMesh) as f:
            for i, line in enumerate(f):
                if i == 1:
                    string = f.readline()
                elif i > 1:
                    break
        temp = re.findall(r'\d+', string)
        res = list(map(int, temp))
        nNodes = res[0]
        nElem = res[1]
        return nNodes, nElem

    def readFiles(self, fnameMesh):
        self.nNodes, self.nElem = generateDataset.obtainNnodesAndElem(fnameMesh)
        self.nodes = np.genfromtxt(fnameMesh, skip_header=3, skip_footer=self.nElem)
        self.elem = np.genfromtxt(fnameMesh, skip_header=3 + self.nNodes, dtype=int)
        return
    
    def compileData(self):
        l = 0
        for k in range(self.ninit, self.nend + self.ngap, self.ngap):
            fnameMesh = os.path.join(self.folder, "marker.{:>07d}.dat".format(k))
            self.readFiles(fnameMesh)
            self.AllNodes[:, :, l] = self.nodes[:, 0:3]
            self.AllVel[:, :, l] = self.nodes[:, 3:6]
            self.AllElem[:, :, l] = self.elem[:, 0:3]
            l += 1

        self.AllElem = np.array(self.AllElem, dtype=int)

        print(self.AllNodes.shape)

    def splitData(self):
        numNodes, coords, ntsteps = self.AllNodes.shape
        num_splits = ntsteps - self.split + 1
        
        self.SplitNodes = np.zeros((num_splits, numNodes, self.split, coords))
        self.SplitVel = np.zeros((num_splits, numNodes, self.split, coords))
        
        for i in range(num_splits):
            self.SplitNodes[i] = self.AllNodes[:, :, i:i+self.split].transpose(0, 2, 1)
            self.SplitVel[i] = self.AllVel[:, :, i:i+self.split].transpose(0, 2, 1)

        
        self.SplitElem = self.AllElem[:, :, 0]

    def get_output_split(self):
        return self.SplitNodes, self.SplitVel, self.SplitElem
    
    def get_output_full(self):
        return self.AllNodes, self.AllVel, self.AllElem

class unstructMeshGenerator():
    def __init__(self,nodes,vel,elem):
        self.nodes = nodes #[ntsteps/batches, nNodes, input-output, features]
        self.elem = elem #[nElem, connections]
        self.vel = vel
        self.nNodes = len(self.nodes[0,:,0,0])
        self.nElem = len(self.elem[:,0])

    def build_grid(self,k):
        return torch.tensor(self.nodes[k,:,0,:], dtype=torch.float32)

    def getEdgeAttr(self,r):
        coords = self.nodes[0,:,0,:]
        pwd = sklearn.metrics.pairwise_distances(coords)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]
        
        return torch.tensor(self.edge_index, dtype=torch.long)
    
    def attributes(self,k):
        edge_attr = np.zeros((self.n_edges, 12))
        for n, (i,j) in enumerate(self.edge_index.transpose()):
            edge_attr[n,:] = np.array([
                                        self.nodes[k,i,0,0], self.nodes[k,i,0,1], self.nodes[k,i,0,2], 
                                        self.nodes[k,j,0,0], self.nodes[k,j,0,1], self.nodes[k,j,0,2],
                                        self.vel[k,i,0,0], self.vel[k,i,0,1], self.vel[k,i,0,2],
                                        self.vel[k,j,0,1], self.vel[k,j,0,1], self.vel[k,j,0,2]
                                        ])

        return torch.tensor(edge_attr, dtype=torch.float)
    
    def getInputOutput(self,k):
        input = np.concatenate((self.nodes[k, :, 0, :], self.vel[k, :, 0, :]), axis=1)
        output = np.concatenate((self.nodes[k, :, 1, :], self.vel[k, :, 1, :]), axis=1)

        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)


class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    
    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd**2/sigma**2)
        sample = np.random.binomial(1,rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)


    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            else:
                edge_attr = np.zeros((self.n_edges, 3*self.d))
                edge_attr[:,0:2*self.d] = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d +1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            if theta is None:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3*self.d))
                edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
            if theta is None:
                edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)
    
class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    
    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd**2/sigma**2)
        sample = np.random.binomial(1,rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)


    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            else:
                edge_attr = np.zeros((self.n_edges, 3*self.d))
                edge_attr[:,0:2*self.d] = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d +1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            if theta is None:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3*self.d))
                edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
            if theta is None:
                edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
                self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid


    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd**2/sigma**2)
        sample = np.random.binomial(1,rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)


    # def get_boundary(self):
    #     s = self.s
    #     n = self.n
    #     boundary1 = np.array(range(0, s))
    #     boundary2 = np.array(range(n - s, n))
    #     boundary3 = np.array(range(s, n, s))
    #     boundary4 = np.array(range(2 * s - 1, n, s))
    #     self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])
    #
    # def boundary_connectivity2d(self, stride=1):
    #
    #     boundary = self.boundary[::stride]
    #     boundary_size = len(boundary)
    #     vertice1 = np.array(range(self.n))
    #     vertice1 = np.repeat(vertice1, boundary_size)
    #     vertice2 = np.tile(boundary, self.n)
    #     self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
    #     self.n_edges_boundary = self.edge_index_boundary.shape[1]
    #     return torch.tensor(self.edge_index_boundary, dtype=torch.long)
    #
    # def attributes_boundary(self, f=None, theta=None):
    #     # if self.edge_index_boundary == None:
    #     #     self.boundary_connectivity2d()
    #     if f is None:
    #         if theta is None:
    #             edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         else:
    #             edge_attr_boundary = np.zeros((self.n_edges_boundary, 3*self.d))
    #             edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #             edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
    #             edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
    #     else:
    #         xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         if theta is None:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
    #         else:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])
    #
    #     return torch.tensor(edge_attr_boundary, dtype=torch.float)

class RandomGridSplitter(object):
    def __init__(self, grid, resolution, m=200, l=2, radius=0.25):
        super(RandomGridSplitter, self).__init__()

        self.grid = grid
        self.resolution = resolution
        self.n = resolution**2
        self.m = m
        self.l = l
        self.radius = radius

        assert self.n % self.m == 0
        self.num = self.n // self.m

    def get_data(self, theta):

        data = []
        for i in range(self.l):
            perm = torch.randperm(self.n)
            perm = perm.reshape(self.num, self.m)

            for j in range(self.num):
                idx = perm[j,:].reshape(-1,)
                grid_sample = self.grid.reshape(self.n,-1)[idx]
                theta_sample = theta.reshape(self.n,-1)[idx]

                X = torch.cat([grid_sample,theta_sample],dim=1)

                pwd = sklearn.metrics.pairwise_distances(grid_sample)
                edge_index = np.vstack(np.where(pwd <= self.radius))
                n_edges = edge_index.shape[1]
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                edge_attr = np.zeros((n_edges, 6))
                a = theta_sample[:,0]
                edge_attr[:, :4] = grid_sample[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4] = a[edge_index[0]]
                edge_attr[:, 5] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.num * self.l // batch_size2

        out = torch.zeros(self.n, )
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i].reshape(batch_size2, self.m)
            for j in range(batch_size2):
                pred_ij = pred_i[j,:].reshape(-1,)
                idx = split_idx_i[j,:].reshape(-1,)
                out[idx] = pred_ij

        out = out / self.l

        # out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        # out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1,)