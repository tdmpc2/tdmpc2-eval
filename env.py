from copy import deepcopy
from collections import defaultdict
import warnings

import gym
import numpy as np
import torch

from envs.dmcontrol import make_env as make_dm_control_env
from envs.maniskill import make_env as make_maniskill_env
from envs.metaworld import make_env as make_metaworld_env
from envs.myosuite import make_env as make_myosuite_env
from envs.exceptions import UnknownTaskError

warnings.filterwarnings('ignore', category=DeprecationWarning)


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		x = torch.from_numpy(x)
		if x.dtype == torch.float64:
			x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info


class MultitaskWrapper(gym.Wrapper):
	"""
	Wrapper for multi-task environments.
	"""

	def __init__(self, cfg, envs):
		super().__init__(envs[0])
		self.cfg = cfg
		self.envs = envs
		self._task = cfg.tasks[0]
		self._task_idx = 0
		self._obs_dims = [env.observation_space.shape[0] for env in self.envs]
		self._action_dims = [env.action_space.shape[0] for env in self.envs]
		self._episode_lengths = [env.max_episode_steps for env in self.envs]
		self._obs_shape = (max(self._obs_dims),)
		self._action_dim = max(self._action_dims)
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
		)
	
	@property
	def task(self):
		return self._task
	
	@property
	def task_idx(self):
		return self._task_idx
	
	@property
	def _env(self):
		return self.envs[self.task_idx]

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _pad_obs(self, obs):
		if obs.shape != self._obs_shape:
			obs = torch.cat((obs, torch.zeros(self._obs_shape[0]-obs.shape[0], dtype=obs.dtype, device=obs.device)))
		return obs
	
	def reset(self, task_idx=-1):
		self._task_idx = task_idx
		self._task = self.cfg.tasks[task_idx]
		self.env = self._env
		return self._pad_obs(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action[:self.env.action_space.shape[0]])
		return self._pad_obs(obs), reward, done, info


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise UnknownTaskError(task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)
	else:
		env = None
		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env]:
			try:
				env = fn(cfg)
			except UnknownTaskError:
				pass
		if env is None:
			raise UnknownTaskError(cfg.task)
		env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {'state': env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	return env
