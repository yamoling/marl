import torch
import numpy as np
from rlenv import Episode
from .batch import Batch


class EpisodeBatch(Batch):
    def __init__(self, episodes: list[Episode], sample_indices: list[int]):
        super().__init__(len(episodes), episodes[0].n_agents, sample_indices)
        self._max_episode_len = max(len(e) for e in episodes)
        self._n_actions = episodes[0].n_actions
        self.episodes = [e.padded(self._max_episode_len) for e in episodes]
        self.masks = torch.tensor(np.array([e.mask for e in episodes])).transpose(1, 0)

    def for_individual_learners(self) -> "EpisodeBatch":
        super().for_individual_learners()
        self.masks = self.masks.repeat_interleave(self.n_agents).view(*self.masks.shape, self.n_agents)
        return self

    def for_rnn(self) -> "Batch":
        """Reshape observations, extras and available actions such that dimensions 1 and 2 are merged (required for GRU)"""
        obs_size = self.obs.shape[3:]
        self.obs = self.obs.reshape(self._max_episode_len, self.size * self.n_agents, *obs_size)
        self.obs_ = self.obs_.reshape(self._max_episode_len, self.size * self.n_agents, *obs_size)
        self.extras = self.extras.reshape(self._max_episode_len, self.size * self.n_agents, -1)
        self.extras_ = self.extras_.reshape(self._max_episode_len, self.size * self.n_agents, -1)
        self.available_actions_ = self.available_actions_.reshape(self._max_episode_len, self.size * self.n_agents, self._n_actions)
        return self

    def _get_obs(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.obs.data for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_obs_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.obs_.data for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_extras(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.extras for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_extras_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.extras_ for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_actions(self) -> torch.LongTensor:
        actions = torch.from_numpy(np.array([e.actions for e in self.episodes], dtype=np.int64))
        return actions.transpose(1, 0).unsqueeze(-1)
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.rewards for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_dones(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.dones for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_available_actions_(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([e.available_actions_ for e in self.episodes], dtype=np.int64)).transpose(1, 0)
    
    def _get_available_actions(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([e.available_actions for e in self.episodes], dtype=np.int64)).transpose(1, 0)
    
    def _get_states(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.states[:-1] for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_states_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.states[1:] for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_action_probs(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.actions_probs for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    