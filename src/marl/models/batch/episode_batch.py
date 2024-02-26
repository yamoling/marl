from functools import cached_property
import torch
import numpy as np
from rlenv import Episode
from .batch import Batch


class EpisodeBatch(Batch):
    def __init__(self, episodes: list[Episode]):
        super().__init__(len(episodes), episodes[0].n_agents)
        self._max_episode_len = max(len(e) for e in episodes)
        self._n_actions = episodes[0].n_actions
        self._base_episides = episodes
        self.episodes = [e.padded(self._max_episode_len) for e in episodes]

    def for_individual_learners(self):
        self.masks = self.masks.repeat_interleave(self.n_agents).view(*self.masks.shape, self.n_agents)
        return super().for_individual_learners()

    def compute_returns(self, gamma: float) -> torch.Tensor:
        result = torch.zeros_like(self.rewards, dtype=torch.float32)
        next_step_returns = self.rewards[-1]
        result[-1] = next_step_returns
        for step in range(self._max_episode_len - 2, -1, -1):
            next_step_returns = self.rewards[step] + gamma * next_step_returns
            result[step] = next_step_returns
        return result

    # def compute_normalized_returns(self, gamma: float, last_obs_value: Optional[float] = None) -> torch.Tensor:
    #     """Compute the returns for each timestep in the batch"""
    #     returns = self.compute_returns(gamma, last_obs_value)
    #     # Normalize the returns such that the algorithm is more stable across environments
    #     # Add 1e-8 to the std to avoid dividing by 0 in case all the returns are equal to 0
    #     normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    #     return normalized_returns

    @cached_property
    def value(self):
        raise NotImplementedError("TODO")

    @cached_property
    def obs_(self):
        """
        All the observations of the episodes, from 0 to episode_length + 1.

        The reason why we have episode_length + 1 is because we need the first observation of the episode
        to perform a correct inference right from the start.
        """
        return torch.from_numpy(np.array([e._observations for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def extras_(self):
        """All the extras of the episodes, from 0 to episode_length + 1"""
        return torch.from_numpy(np.array([e._extras for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def obs(self):
        obs = np.array([e.obs.data for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(obs).transpose(1, 0).to(self.device)

    @cached_property
    def extras(self):
        return torch.from_numpy(np.array([e.extras for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def actions(self):
        actions = torch.from_numpy(np.array([e.actions for e in self.episodes], dtype=np.int64)).to(self.device)
        return actions.unsqueeze(-1).transpose(1, 0)

    @cached_property
    def rewards(self):
        return torch.from_numpy(np.array([e.rewards for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def dones(self):
        return torch.from_numpy(np.array([e.dones for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def available_actions_(self):
        return torch.from_numpy(np.array([e.available_actions_ for e in self.episodes], dtype=np.int64)).transpose(1, 0).to(self.device)

    @cached_property
    def available_actions(self):
        return torch.from_numpy(np.array([e.available_actions for e in self.episodes], dtype=np.int64)).transpose(1, 0).to(self.device)

    @cached_property
    def states(self):
        return torch.from_numpy(np.array([e.states[:-1] for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def states_(self):
        return torch.from_numpy(np.array([e.states[1:] for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)

    @cached_property
    def masks(self):
        return torch.from_numpy(np.array([e.mask for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)
