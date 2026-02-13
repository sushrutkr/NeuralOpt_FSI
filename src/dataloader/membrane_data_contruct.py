import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import BallTree
import os
import re
import matplotlib.pyplot as plt

class generateDatasetMembrane:
  def __init__(self, ninit, nend, ngap, splitLen=1, folder="./"):
    self.nodes = []
    self.elem = []
    self.ninit = ninit
    self.nend = nend 
    self.ngap = ngap
    self.split = splitLen
    self.ntsteps = int(((nend - ninit) / ngap) + 1)
    assert self.ntsteps - splitLen > 0
    self.folder = folder
    fnameMesh = os.path.join(folder, "marker.{:>07d}.dat".format(ninit))
    self.nNodes, self.nElem = generateDatasetMembrane.obtainNnodesAndElem(fnameMesh)
    self.AllNodes = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllVel = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllPressure = np.zeros(shape=(self.nNodes, self.ntsteps))
    self.AllForce = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllElem = np.zeros(shape=(self.nElem, 3, self.ntsteps))
    self.SplitNodes = np.array([])
    self.SplitVel = np.array([])
    self.SplitElem = np.array([])
    self.BCNodes = self.get_bc_node_info()
    self.compileData()
    self.calculate_point_mass(thickness=0.02,density=1)

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
    self.nNodes, self.nElem = generateDatasetMembrane.obtainNnodesAndElem(fnameMesh)
    self.nodes = np.genfromtxt(fnameMesh, skip_header=3, skip_footer=self.nElem)
    self.elem = np.genfromtxt(fnameMesh, skip_header=3 + self.nNodes, dtype=int)
    return
  
  def calculate_point_mass(self, thickness, density):
    self.pointMass = np.zeros(shape=(self.nNodes))

    edges_AB = self.nodes[self.elem[:,0]-1,0:3] - self.nodes[self.elem[:,1]-1,0:3]
    edges_AC = self.nodes[self.elem[:,0]-1,0:3] - self.nodes[self.elem[:,2]-1,0:3]
    normal = np.cross(edges_AB, edges_AC)/2
    area = np.linalg.norm(normal, axis=1)

    for iElem in range(self.nElem):
      mass = area[iElem]*thickness*density
      for i in range(3):
        nodeIndex = self.elem[iElem,i] - 1
        self.pointMass[nodeIndex] += mass/3
    
    return
  
  def get_bc_node_info(self):
    nodeIDs = np.genfromtxt(self.folder+"../../"+"dirichlet_bc_nodes.csv", skip_header=1)
    return nodeIDs
  
  def compileData(self):
    l = 0
    for k in range(self.ninit, self.nend + self.ngap, self.ngap):
      fnameMesh = os.path.join(self.folder, "marker.{:>07d}.dat".format(k))
      self.readFiles(fnameMesh)
      self.AllNodes[:, :, l] = self.nodes[:, 0:3]
      self.AllVel[:, :, l] = self.nodes[:, 3:6]
      self.AllPressure[:, l] = self.nodes[:, 6]
      self.AllForce[:, :, l] = self.nodes[:, 7:10]
      self.AllElem[:, :, l] = self.elem[:, 0:3]
      l += 1

    self.AllElem = np.array(self.AllElem, dtype=int)

  def scaling(self,scaler):
    # scaling node coords
    # Although I have max and min coords available I will not be using it because that disturb the scaling of flow coords
    self.AllNodes[:, :, :] -= 20
    
    # scaling velocity
    self.AllVel = self.AllVel/scaler["velocity_scale"]

    # scaling pressure
    self.AllPressure = self.AllPressure/scaler["pressure_scale"]

    # scaling force
    self.AllForce = self.AllForce/scaler["force_scale"]

    # scaling pointMass
    self.pointMass = (self.pointMass - scaler["pointMass_min"])/(scaler["pointMass_max"] - scaler["pointMass_min"])

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i in range(3):
      axes[0, i].hist(self.AllNodes[:,i,:].flatten(), bins=500, density=True, alpha=0.7, color='blue')
      axes[0, i].set_title("Membrane X - %i"%(i+1))
    for i in range(3):
      axes[1, i].hist(self.AllVel[:,i,:].flatten(), bins=500, density=True, alpha=0.7, color='blue')
      axes[1, i].set_title("Membrane U - %i"%(i+1))
    for i in range(3):
      axes[2, i].hist(self.AllForce[:,i,:].flatten(), bins=500, density=True, alpha=0.7, color='blue')
      axes[2, i].set_title("Membrane F - %i"%(i+1))
    axes[3, 0].hist(self.AllPressure.flatten(), bins=500, density=True, alpha=0.7, color='blue')
    axes[3, 0].set_title("Membrane P")
    axes[3, 1].hist(self.pointMass.flatten(), bins=500, density=True, alpha=0.7, color='blue')
    axes[3, 1].set_title("Membrane mass")
    axes[3, 2].axis('off')
    plt.tight_layout()
    plt.savefig("./logs/membrane_hist.png")
    plt.close()

  def splitData(self):
    numNodes, coords, ntsteps = self.AllNodes.shape
    num_splits = ntsteps - self.split + 1
    
    self.SplitNodes = np.zeros((num_splits, numNodes, self.split, coords))
    self.SplitVel = np.zeros((num_splits, numNodes, self.split, coords))
    self.SplitPressure = np.zeros((num_splits, numNodes, self.split))
    self.SplitExtForce = np.zeros((num_splits, numNodes, self.split, coords))
    
    for i in range(num_splits):
      self.SplitNodes[i] = self.AllNodes[:, :, i:i+self.split].transpose(0, 2, 1)
      self.SplitVel[i] = self.AllVel[:, :, i:i+self.split].transpose(0, 2, 1)
      self.SplitPressure[i] = self.AllPressure[:, i:i+self.split].transpose(0, 1)
      self.SplitExtForce[i] = self.AllForce[:, :, i:i+self.split].transpose(0, 2, 1)
    
    self.SplitElem = self.AllElem[:, :, 0]

  def get_output_split(self):
    return self.SplitNodes, self.SplitVel, self.SplitPressure, self.SplitExtForce, self.SplitElem, np.expand_dims(self.pointMass, axis=1), self.BCNodes
  
  def get_output_full(self):
    return self.AllNodes, self.AllVel, self.AllPressure, self.AllForce, np.expand_dims(self.pointMass, axis=1), self.AllElem

class unstructMeshGenerator():
  def __init__(self,nodes,vel,pressure,forceExt,pointMass,elem,bc_nodes):
    self.nodes = nodes #[ntsteps/batches, nNodes, input-output, features]
    self.elem = elem #[nElem, connections]
    self.vel = vel
    self.pressure = pressure
    self.forceExt = forceExt
    self.pointMass = pointMass
    self.nNodes = len(self.nodes[0,:,0,0])
    self.nElem = len(self.elem[:,0])
    self.bc_nodes = bc_nodes.flatten().astype(int)

  def build_grid(self,k):
    return self.nodes[k,:,0,:]

  def getEdgeAttr(self,r):
    coords = self.nodes[0,:,0,:]
    pwd = sklearn.metrics.pairwise_distances(coords)
    self.edge_index = np.vstack(np.where(pwd <= r))
    self.n_edges = self.edge_index.shape[1]
    
    return self.edge_index
  
  def attributes(self,k):
    #shape of node data (e.g. coords) - (ntsteps, nNodes, input-output(s), vector component)
    edge_attr = np.zeros((self.n_edges, 7))
    for n, (i,j) in enumerate(self.edge_index.transpose()):
        edge_attr[n,:] = np.array([
                                    self.nodes[k,i,0,0] - self.nodes[k,j,0,0], 
                                    self.nodes[k,i,0,1] - self.nodes[k,j,0,1], 
                                    self.nodes[k,i,0,2] - self.nodes[k,j,0,2],
                                    np.sqrt((self.nodes[k,i,0,0] - self.nodes[k,j,0,0])**2 +
                                            (self.nodes[k,i,0,1] - self.nodes[k,j,0,1])**2 +
                                            (self.nodes[k,i,0,2] - self.nodes[k,j,0,2])**2),
                                    self.vel[k,i,0,0] - self.vel[k,j,0,0], 
                                    self.vel[k,i,0,1] - self.vel[k,j,0,1], 
                                    self.vel[k,i,0,2] - self.vel[k,j,0,2]
                                    ])

    return edge_attr
  
  def getInputOutput(self,k):
    input = np.concatenate((self.nodes[k, :, 0, :], 
                            self.vel[k, :, 0, :], 
                            np.expand_dims(self.pressure[k,:,0], axis=1),
                            self.forceExt[k, :, 0, :],
                            self.pointMass[:]),
                            axis=1)
    
    output = np.concatenate((self.nodes[k, :, 1:, :], 
                             self.vel[k, :, 1:, :], 
                             np.expand_dims(self.pressure[k,:,1:], axis=2),
                             self.forceExt[k, :, 1:, :]), axis=2) #1: is to extract next timestep data
    output = np.transpose(output, (1, 0, 2))

    bc = np.ones(shape=(self.nNodes,7)) # 7 - [flag, 3 coords, 3 vels], no need for force as that's going to come from flow
    bc[self.bc_nodes,0] = 0 #to nullify any preditions at those nodes
    #--- Since the dimen for output is output shape :  (1, 1757, 9), i have to use first index otherwise it shoudl not be use
    bc[self.bc_nodes,1:] = output[0,self.bc_nodes,0:6]

    return input, output, bc