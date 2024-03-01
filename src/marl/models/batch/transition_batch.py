from functools import cached_property
import torch
import numpy as np
from rlenv import Transition

from .batch import Batch


class TransitionBatch(Batch):
    def __init__(self, transitions: list[Transition]):
        super().__init__(len(transitions), transitions[0].n_agents)
        self.transitions = transitions

    @cached_property
    def obs(self):
        return torch.from_numpy(np.array([t.obs.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def obs_(self):
        return torch.from_numpy(np.array([t.obs_.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def extras(self):
        return torch.from_numpy(np.array([t.obs.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def extras_(self):
        return torch.from_numpy(np.array([t.obs_.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def actions(self):
        return torch.from_numpy(np.array([t.action for t in self.transitions], dtype=np.int64)).unsqueeze(-1).to(self.device)

    @cached_property
    def rewards(self):
        return torch.from_numpy(np.array([t.reward for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def dones(self):
        return torch.from_numpy(np.array([t.done for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def available_actions(self):
        return torch.from_numpy(np.array([t.obs.available_actions for t in self.transitions], dtype=np.int64)).to(self.device)

    @cached_property
    def available_actions_(self):
        return torch.from_numpy(np.array([t.obs_.available_actions for t in self.transitions], dtype=np.int64)).to(self.device)

    @cached_property
    def states(self):
        return torch.from_numpy(np.array([t.obs.state for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def states_(self):
        return torch.from_numpy(np.array([t.obs_.state for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def masks(self):
        return torch.ones(self.size).to(self.device)
