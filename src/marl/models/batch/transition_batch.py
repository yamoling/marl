from functools import cached_property
import torch
import numpy as np
from rlenv import Transition

from .batch import Batch


class TransitionBatch(Batch):
    def __init__(
            self, transitions: list[Transition], sample_indices: list[int]):
        super().__init__(len(transitions), transitions[0].n_agents, sample_indices)
        self.transitions = transitions

    @cached_property
    def obs(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def obs_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs_.data for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def extras(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs.extras for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def extras_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs_.extras for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def actions(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([t.action for t in self.transitions], dtype=np.int64)).unsqueeze(-1).to(self.device)
    
    @cached_property
    def rewards(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.reward for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def dones(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.done for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def available_actions(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([t.obs.available_actions for t in self.transitions], dtype=np.int64)).to(self.device)
    
    @cached_property
    def available_actions_(self) -> torch.LongTensor:
        return torch.from_numpy(np.array([t.obs_.available_actions for t in self.transitions], dtype=np.int64)).to(self.device)
    
    @cached_property
    def states(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs.state for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def states_(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.obs_.state for t in self.transitions], dtype=np.float32)).to(self.device)
    
    @cached_property
    def action_probs(self) -> torch.Tensor:
        return torch.from_numpy(np.array([t.action_prob for t in self.transitions], dtype=np.float32)).to(self.device)
