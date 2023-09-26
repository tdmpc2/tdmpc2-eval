from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import helper as h


class WorldModel(nn.Module):
	"""TD-MPC2 implicit world model."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=h.SimNorm(cfg))
		self._reward = h.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = h.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = h.Ensemble([h.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.log_std_min = torch.tensor(cfg.log_std_min)
		self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		if self.cfg.multitask:
			self._action_masks = self._action_masks.to(*args, **kwargs)
		self.log_std_min = self.log_std_min.to(*args, **kwargs)
		self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
		return self
	
	def task_emb(self, x, task):
		"""Continuous task embedding for multi-task experiments."""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""Encodes an observation into its latent representation (h)."""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		return self._encoder['state'](obs)

	def next(self, z, a, task):
		"""Predicts next latent state (d) and single-step reward (R)."""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)
	
	def reward(self, z, a, task):
		"""Predicts single-step reward (R)."""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def pi(self, z, task):
		"""Samples an action from the learned policy (pi)."""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = h.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)

		if self.cfg.multitask:
			mu = mu * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]

		pi = mu + eps * log_std.exp()
		return torch.tanh(mu), torch.tanh(pi)

	def Q(self, z, a, task):
		"""Predict state-action value (Q)."""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		out = self._Qs(z)

		Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
		Q1, Q2 = h.two_hot_inv(Q1, self.cfg), h.two_hot_inv(Q2, self.cfg)
		return (Q1 + Q2) / 2


class TDMPC2:
	"""Minimal implementation of TD-MPC2 model inference."""

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.model_target = deepcopy(self.model)
		self.model.eval()
		self.model_target.eval()
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		print("Total parameters: {:,}".format(
			sum(p.numel() for p in self.model.parameters())))
		self.gamma = [self.get_gamma(ep_len) for ep_len in cfg.episode_lengths] \
			if self.cfg.multitask else self.get_gamma(cfg.episode_length)

	def get_gamma(self, episode_length):
		"""
		Returns gamma for a given episode length.
		Simple heuristic that scales gamma linearly with episode length.
		"""
		return min(max(((episode_length/5)-1)/(episode_length/5), 0.95), 0.995)

	def load(self, fp):
		"""Load a saved state dict from filepath (or dictionary) into current agent."""
		d = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(d["model"])
		self.model_target.load_state_dict(d["model_target"])

	@torch.no_grad()
	def act(self, obs, t0=False, task=None):
		"""Take an action. Uses either MPC or the learned policy, depending on the self.cfg.mpc flag."""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		a = self.plan(z, t0=t0, task=task)
		return a.cpu()

	@torch.no_grad()
	def estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = h.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G += discount * reward
			discount *= self.gamma[torch.tensor(task)] if self.cfg.multitask else self.gamma
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task)

	@torch.no_grad()
	def plan(self, z, t0=False, task=None):
		"""Plan a sequence of actions using the learned world model."""		
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]
			self._pi_mean = pi_actions.mean(1)
			self._pi_std = pi_actions.std(1)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for i in range(self.cfg.iterations):

			# Sample actions
			actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self.estimate_value(z, actions, task).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		return actions[0].clamp_(-1, 1)
