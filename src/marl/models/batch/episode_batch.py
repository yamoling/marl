from functools import cached_property
import torch
import numpy as np
from rlenv import Episode
from .batch import Batch


class EpisodeBatch(Batch):
    def __init__(self, episodes: list[Episode], sample_indices: list[int]):
        super().__init__(len(episodes), episodes[0].n_agents, sample_indices)
        self._max_episode_len = max(len(e) for e in episodes)
        self._n_actions = episodes[0].n_actions
        self._base_episides = episodes
        self.episodes = [e.padded(self._max_episode_len) for e in episodes]
        self.masks = torch.from_numpy(np.array([e.mask for e in self.episodes], dtype=np.float32)).transpose(1, 0)

    def for_individual_learners(self) -> "EpisodeBatch":
        super().for_individual_learners()
        self.masks = self.masks.repeat_interleave(self.n_agents).view(*self.masks.shape, self.n_agents)
        return self

    @cached_property
    def obs_(self):
        """All the observations of the episodes, from 0 to episode_length + 1"""
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
    def action_probs(self):
        return torch.from_numpy(np.array([e.actions_probs for e in self.episodes], dtype=np.float32)).transpose(1, 0).to(self.device)
