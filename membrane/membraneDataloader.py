import numpy as np
import torch
import os
import re

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

    def splitData(self):
        numNodes, coords, ntsteps = self.AllNodes.shape
        num_splits = ntsteps - self.split + 1
        
        self.SplitNodes = np.zeros((num_splits, numNodes, self.split, coords))
        self.SplitVel = np.zeros((num_splits, numNodes, self.split, coords))
        
        for i in range(num_splits):
            self.SplitNodes[i] = self.AllNodes[:, :, i:i+self.split].transpose(0, 2, 1)
            self.SplitVel[i] = self.AllVel[:, :, i:i+self.split].transpose(0, 2, 1)

        
        self.SplitElem = self.AllElem[:, :, 0]

    def get_output(self):
        return self.SplitNodes, self.SplitVel, self.SplitElem

class MembraneDataset:
    def __init__(self, partition, max_samples, delta_frame, data_dir="./", num_timesteps=21):
        self.partition = partition
        self.max_samples = max_samples
        self.delta_frame = delta_frame
        self.data_dir = data_dir  # Convert to absolute path
        self.num_timesteps = num_timesteps
        
        # Read Tecplot data
        obtainDataset = generateDataset(ninit=3000, nend=4000, ngap=50, splitLen=4, folder=self.data_dir)
        nodes, vel, connectivity = obtainDataset.get_output()
        nodes -= 20
        connectivity -= 1

        print(f"Dataset initialized with {nodes.shape[1]} nodes and {connectivity.shape[0]} elements.")
        np.save("connectivity.npy",connectivity)
        
        # Process data
        self.process_data(nodes, vel, connectivity)
        
    def process_data(self, nodes, vel, connectivity):
        N = nodes.shape[1]
        self.n_node = N
        
        # Create edge list from connectivity
        edges = []
        for conn in connectivity:
            for i in range(len(conn)):
                for j in range(i + 1, len(conn)):
                    if conn[i] < N and conn[j] < N:
                        edges.append((conn[i], conn[j]))
                    else:
                        print(f"Warning: Edge ({conn[i]}, {conn[j]}) is out of bounds for atom_edges with size {N}")
        
        # Initialize edge attributes
        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1
        
        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        
        edge_attr = []
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                    elif self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
        
        self.edges = torch.LongTensor(np.array([rows, cols]))
        self.edge_attr = torch.Tensor(np.array(edge_attr))
        
        # Split data into train, val, test
        self.split_data(nodes, vel)
        
    def split_data(self, nodes, vel):
        N = nodes.shape[1]
        x_0, v_0, x_t, v_t = [], [], [], []

        for i in range(self.max_samples):
            cur_x_0 = nodes[i,:,0,:]
            cur_v_0 = vel[i,:,0,:]
            cur_x_t = nodes[i,:,:,:]
            cur_v_t = vel[i,:,:,:]
            
            x_0.append(cur_x_0)
            v_0.append(cur_v_0)
            x_t.append(cur_x_t)
            v_t.append(cur_v_t)
        
        # Convert lists of numpy arrays to numpy arrays before converting to tensors
        self.x_0 = torch.from_numpy(np.array(x_0)).float()
        self.v_0 = torch.from_numpy(np.array(v_0)).float()
        self.x_t = torch.from_numpy(np.array(x_t)).float()
        self.v_t = torch.from_numpy(np.array(v_t)).float()
        self.mole_idx = torch.tensor(np.ones(N))
    
    def __getitem__(self, i):
        edges = self.edges
        edge_attr = self.edge_attr
        local_edge_mask = edge_attr[..., -1] == 1
        local_edges = edges[..., local_edge_mask]
        local_edge_attr = edge_attr[local_edge_mask]
        
        node_fea = self.x_0[i][..., 1].unsqueeze(-1) / 10
        return self.x_0[i], self.v_0[i], edges, edge_attr, local_edges, local_edge_attr, node_fea, self.x_t[i], self.v_t[i]
    
    def __len__(self):
        length = len(self.x_0)
        return length

if __name__ == '__main__':
    data = MembraneDataset(partition='train', max_samples=200, delta_frame=1, data_dir='/home/skumar94/Desktop/EGNO/membrane/1e6/', num_timesteps=18)
    print(data[:][-1].shape, data[:][0].shape)
    # print(data[2][6])

    dataset_train = MembraneDataset(partition='train', max_samples=18, data_dir='/home/skumar94/Desktop/EGNO/membrane/1e6/',
                                    delta_frame=1, num_timesteps=18)
    print(dataset_train[:][-1].shape, dataset_train[:][0].shape)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=18, shuffle=True)
    print(f"Training DataLoader created with {len(loader_train)} batches.")