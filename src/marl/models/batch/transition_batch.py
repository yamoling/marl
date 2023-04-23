import torch
import numpy as np
from rlenv import Transition

from .batch import Batch


class TransitionsBatch(Batch):
    def __init__(self, transitions: list[Transition]):
        super().__init__(len(transitions), transitions[0].n_agents)
        self.transitions = transitions

    def _get_obs(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs.data for t in self.transitions], dtype=np.float32))

    def _get_obs_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs_.data for t in self.transitions], dtype=np.float32))
    
    def _get_extras(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs.extras for t in self.transitions], dtype=np.float32))
    
    def _get_extras_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs_.extras for t in self.transitions], dtype=np.float32))
    
    def _get_actions(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([t.action for t in self.transitions], dtype=np.int64)).unsqueeze(-1)
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.reward for t in self.transitions], dtype=np.float32))
    
    def _get_dones(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.done for t in self.transitions], dtype=np.float32))
    
    def _get_available_actions(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([t.obs.available_actions for t in self.transitions], dtype=np.int64))
    
    def _get_available_actions_(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([t.obs_.available_actions for t in self.transitions], dtype=np.int64))
    
    def _get_states(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs.state for t in self.transitions], dtype=np.float32))
    
    def _get_states_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs_.state for t in self.transitions], dtype=np.float32))
    
    def _get_action_probs(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.action_prob for t in self.transitions], dtype=np.float32))
