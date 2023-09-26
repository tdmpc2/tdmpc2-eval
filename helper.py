import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble

DREG_BINS = None


@torch.jit.script
def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.size(0), cfg.num_bins, device=x.device)
	soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	global DREG_BINS
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	if DREG_BINS is None:
		DREG_BINS = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
	return symexp(x)


def pcount(m):
	count = sum(p.numel() for p in m.parameters())
	return f"{count:,}"


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		modules = nn.ModuleList(modules)
		fn, params, _ = combine_state_for_ensemble(modules)
		self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different', **kwargs)
		self.params = nn.ParameterList([nn.Parameter(p) for p in params])
		self._repr = str(modules)

	def modules(self):
		return self.vmap.__wrapped__.stateless_model

	def forward(self, *args, **kwargs):
		return self.vmap([p for p in self.params], (), *args, **kwargs)

	def __repr__(self):
		return 'Vectorized ' + self._repr


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, Mish activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))
	
	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, \
			out_features={self.out_features}, \
			bias={self.bias is not None}{repr_dropout}, \
			act={self.act.__class__.__name__})"


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k,v in cfg.obs_shape.items():
		assert k == 'state'
		out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	return nn.ModuleDict(out)


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)
