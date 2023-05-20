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

    @property
    def obs_(self) -> torch.Tensor:
        """All the observations of the episodes, from 0 to episode_length + 1"""
        return super().obs_

    @property
    def extras_(self) -> torch.Tensor:
        """All the extras of the episodes, from 0 to episode_length + 1"""
        return super().extras_

    def _get_obs(self) -> torch.Tensor:
        try:
            obs = np.array([e.obs.data for e in self.episodes], dtype=np.float32)
        except ValueError as e:
            shapes = [e.obs.data.shape for e in self.episodes]
            lengths = [len(e) for e in self.episodes]
            from pprint import pprint
            print("Shaped:")
            pprint(shapes)
            print("Lengths:")
            pprint(lengths)
            raise e
        return torch.from_numpy(obs).transpose(1, 0)
    
    def _get_obs_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e._observations for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_extras(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e.extras for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_extras_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([e._extras for e in self.episodes], dtype=np.float32)).transpose(1, 0)
    
    def _get_actions(self) -> torch.LongTensor:
        actions = torch.from_numpy(np.array([e.actions for e in self.episodes], dtype=np.int64))
        return actions.unsqueeze(-1).transpose(1, 0)
    
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
    