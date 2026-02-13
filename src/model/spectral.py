import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_timestep_embedding(t, d, structure):
	assert d%2 == 0
	# embeddings
	# sin(2 \pi (1/10^(i/d)) t)
	# cos(2 \pi (1/10^(i/d)) t)
	t = t.view(-1,1).to(device)
	scales = torch.arange(0, d//2).to(device)
	freqs = 10**(scales/(d//2))

	if structure == "ordered":
		emb = 2 * torch.pi*t/freqs
		emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=-1)
	elif structure == "alternating":
		emb = torch.zeros(len(t), d, device=t.device)
		emb[:, 0::2] = torch.sin(2 * torch.pi * t.unsqueeze(-1) / freqs)
		emb[:, 1::2] = torch.cos(2 * torch.pi * t.unsqueeze(-1) / freqs)

	return emb

class time_conditioning(nn.Module):
	def __init__(self, hidden_dim):
		super(time_conditioning, self).__init__()

		self.gamma = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), 
														   nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            									 nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
															 )
		
		self.gamma_residual = nn.Linear(1, hidden_dim)

		self.lambda_ = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), 
															   nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            										 nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
																 )
		
		self.lambda_residual = nn.Linear(1, hidden_dim)

	def forward(self, x, tau):
		x = F.layer_norm(x, x.shape[1:]) #1: because layer_norm requires tu[]:plesof int and not int.
		gamma_tau = self.gamma(tau.view(-1, 1)) + self.gamma_residual(tau.view(-1, 1))
		lambda_tau = self.lambda_(tau.view(-1, 1)) + self.lambda_residual(tau.view(-1, 1))
		x = (1 + tau * gamma_tau) * x + tau * lambda_tau	
		#Process - Scale							= shift

		return x
	

# def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
#     half_dim = embedding_dim // 2
#     emb = math.log(max_positions) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
#     emb = timesteps.float()[:, None] * emb[None, :]
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
#     if embedding_dim % 2 == 1:  # zero pad
#         emb = F.pad(emb, (0, 1), mode='constant')
#     assert emb.shape == (timesteps.shape[0], embedding_dim)
#     return emb


# def variance_scaling(scale, mode, distribution,
#                      in_axis=1, out_axis=0,
#                      dtype=torch.float32,
#                      device='cpu'):
#     """Ported from JAX. """

#     def _compute_fans(shape, in_axis=1, out_axis=0):
#         receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
#         fan_in = shape[in_axis] * receptive_field_size
#         fan_out = shape[out_axis] * receptive_field_size
#         return fan_in, fan_out

#     def init(shape, dtype=dtype, device=device):
#         fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
#         if mode == "fan_in":
#             denominator = fan_in
#         elif mode == "fan_out":
#             denominator = fan_out
#         elif mode == "fan_avg":
#             denominator = (fan_in + fan_out) / 2
#         else:
#             raise ValueError(
#                 "invalid mode for variance scaling initializer: {}".format(mode))
#         variance = scale / denominator
#         if distribution == "normal":
#             return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
#         elif distribution == "uniform":
#             return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
#         else:
#             raise ValueError("invalid distribution for variance scaling initializer")

#     return init


# def default_init(scale=1.):
#     """The same initialization used in DDPM."""
#     scale = 1e-10 if scale == 0 else scale
#     return variance_scaling(scale, 'fan_avg', 'uniform')


# class NIN(nn.Module):
#     def __init__(self, in_dim, num_units, init_scale=0.1):
#         super().__init__()
#         self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
#         self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         y = torch.einsum('bchw,co->bohw', x, self.W) + self.b[None, :, None, None]
#         if y.stride()[1] == 1:
#             y = y.contiguous()
#         return y
	

# @torch.jit.script
# def compl_mul1d(a, b):    
#     # (M, N, in_ch), (in_ch, out_ch, M) -> (M, N, out_channel)
#     return torch.einsum("mni,iom->mno", a, b)


# class SpectralConv1d(nn.Module):
#     def __init__(self, in_ch, out_ch, modes1):
#         super(SpectralConv1d, self).__init__()

#         """
#         1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """

#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.modes1 = modes1

#         self.scale = (1 / (in_ch*out_ch))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))

#     def forward(self, x):
#         T, N, C = x.shape
#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         with torch.cuda.amp.autocast(enabled=False):
#         # with torch.autocast(device_type='cuda', enabled=False):
#             x_ft = torch.fft.rfftn(x.float(), dim=[0])
#             # Multiply relevant Fourier modes
#             out_ft = compl_mul1d(x_ft[:self.modes1], torch.view_as_complex(self.weights1))
#             # Return to physical space
#             x = torch.fft.irfftn(out_ft, s=[T], dim=[0])
#         return x


# class TimeConv(nn.Module):
#     """
#     1. time_conv
#     Purpose: This convolution operation is applied to the hidden features h of the nodes.
#     Input Shape: [T, num_nodes, hidden_nf]
#     Operation: The time_conv module processes the hidden features over time, capturing temporal dependencies and interactions between the nodes' features.
#     Output Shape: The output is reshaped back to [T * num_nodes, hidden_nf] after the convolution.
#     """
#     def __init__(self, in_ch, out_ch, modes, act, with_nin=False):
#         super(TimeConv, self).__init__()
#         self.with_nin = with_nin
#         self.t_conv = SpectralConv1d(in_ch, out_ch, modes)
#         # if with_nin:
#         #     self.nin = NIN(in_ch, out_ch)
#         self.act = nn.LeakyReLU()

#     def forward(self, x):
#         h = self.t_conv(x)
#         # if self.with_nin:
#         #     x = self.nin(x)
#         out = self.act(h)
#         return x + out


# @torch.jit.script
# def compl_mul1d_x(a, b):
#     # (M, N, in_ch), (in_ch, out_ch, M) -> (M, N, out_channel)
#     return torch.einsum("mndi,iom->mndo", a, b)


# class SpectralConv1d_x(nn.Module):
#     def __init__(self, in_ch, out_ch, modes1):
#         super(SpectralConv1d_x, self).__init__()

#         """
#         1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """

#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.modes1 = modes1

#         # self.scale = (1 / (in_ch*out_ch))
#         self.scale = 0.1
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))

#     def forward(self, x):
#         T, N, D, C = x.shape  # D should be 3
#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         with torch.cuda.amp.autocast(enabled=False):
#         # with torch.autocast(device_type='cuda', enabled=False):
#             x_ft = torch.fft.rfftn(x.float(), dim=[0])
#             # Multiply relevant Fourier modes
#             out_ft = compl_mul1d_x(x_ft[:self.modes1], torch.view_as_complex(self.weights1))
#             # Return to physical space
#             x = torch.fft.irfftn(out_ft, s=[T], dim=[0])
#         return x


# class TimeConv_x(nn.Module):
#     """
#     Purpose: This convolution operation is applied to the spatial coordinates and velocities of the nodes.
#     Input Shape: [T, num_nodes, 3, 2]
#     The shape [3, 2] indicates that for each node, there are 3 spatial dimensions (x, y, z) and 2 types of data (coordinates and velocities).
#     Operation: The time_conv_x module processes the spatial and velocity data over time, capturing temporal dependencies and interactions between the nodes' positions and velocities.
#     Output Shape: The output is split into updated coordinates and velocities, which are then reshaped back to [T * num_nodes, 3]
#     """
#     def __init__(self, in_ch, out_ch, modes, act, with_nin=False):
#         super(TimeConv_x, self).__init__()
#         self.with_nin = with_nin
#         self.t_conv = SpectralConv1d_x(in_ch, out_ch, modes)
#         # if with_nin:
#         #     self.nin = NIN(in_ch, out_ch)

#     def forward(self, x):  # x: [T, N, D, C]
#         # x = x.unsqueeze(-1)  # [T, N, D, C]
#         h = self.t_conv(x)
#         # if self.with_nin:
#         #     x = self.nin(x)
#         return x + h


# """
# The primary differences between TimeConv and TimeConv_x classes in the provided code are as follows:

# Input Shape
# TimeConv: The input tensor x has a shape of [T, N, C], where T is the number of timesteps, N is the batch size, and C is the number of channels.
# TimeConv_x: The input tensor x has a shape of [T, N, D, C], where T is the number of timesteps, N is the batch size, D is an additional dimension (likely spatial), and C is the number of channels.
# Spectral Convolution
# TimeConv: Uses the SpectralConv1d class for spectral convolution.
# TimeConv_x: Uses the SpectralConv1d_x class for spectral convolution.
# Complex Multiplication
# TimeConv: Uses the compl_mul1d function for complex multiplication.
# TimeConv_x: Uses the compl_mul1d_x function for complex multiplication, which handles an additional dimension D.
# Activation Function
# TimeConv: Applies a LeakyReLU activation function to the output of the spectral convolution.
# TimeConv_x: Does not explicitly apply an activation function to the output of the spectral convolution.
# Optional NIN Layer
# TimeConv: Contains commented-out code for an optional Network-In-Network (NIN) layer.
# TimeConv_x: Also contains commented-out code for an optional NIN layer, but the implementation is slightly different due to the additional dimension D.
# """