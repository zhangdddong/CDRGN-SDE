import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .mgcn_layer import MGCNConvLayer


class MGCNLayerWrapper(torch.nn.Module):
	noise_type = 'general'
	sde_type = 'ito'

	def __init__(self, batch_size, d_dim, brownian_size, param, num_e):
		super(MGCNLayerWrapper, self).__init__()
		
		self.batch_size = batch_size
		self.d_dim = d_dim
		self.brownian_size = brownian_size
		self.p = param
		self.num_e = num_e

		self.mu = torch.nn.Linear(d_dim, d_dim)
		self.sigma = torch.nn.Linear(d_dim, d_dim * brownian_size)

		if self.p.activation.lower() == 'tanh':
			self.act = torch.tanh
		elif self.p.activation.lower() == 'relu':
			self.act = F.relu
		elif self.p.activation.lower() == 'leakyrelu':
			self.act = F.leaky_relu

		self.edge_index = None
		self.edge_type = None
		# residual layer
		if self.p.res:
			self.res = torch.nn.Parameter(torch.FloatTensor([0.1]))
		# define MGCN Layer
		self.conv1 = MGCNConvLayer(self.p.n_hidden, self.p.n_hidden, act=self.act, params=self.p)
		self.conv2 = MGCNConvLayer(self.p.n_hidden, self.p.n_hidden, act=self.act, params=self.p) if self.p.core_layer == 2 else None
		self.drop_l1 = torch.nn.Dropout(self.p.sde_dropout_1)
		self.drop_l2 = torch.nn.Dropout(self.p.sde_dropout_2)

	def set_graph(self, edge_index, edge_type):
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.edge_index = self.edge_index.to('cuda')
		self.edge_type = self.edge_type.to('cuda')

	def forward_base(self, emb):
		if self.p.res:
			emb = emb + self.res * self.conv1(emb, self.edge_index, self.edge_type, self.num_e)
			emb = self.drop_l1(emb)
			emb = (emb + self.res * self.conv2(emb, self.edge_index, self.edge_type, self.num_e)) if self.p.core_layer == 2 else emb
			emb = self.drop_l2(emb) if self.p.core_layer == 2 else emb
		else:
			emb = self.conv1(emb, self.edge_index, self.edge_type, self.num_e)
			emb = self.drop_l1(emb)
			emb = self.conv2(emb, self.edge_index, self.edge_type, self.num_e) if self.p.core_layer == 2 else emb
			emb = self.drop_l2(emb) if self.p.core_layer == 2 else emb
		return emb

	# Drift
	def f(self, t, y):
		emb = self.forward_base(y)
		return self.mu(emb)  # shape (batch_size, d_dim)

	# Diffusion
	def g(self, t, y):
		return self.sigma(y).view(self.batch_size, self.d_dim, self.brownian_size)
