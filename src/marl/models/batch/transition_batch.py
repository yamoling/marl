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
    def all_obs(self):
        observations = [self.transitions[0].obs.data] + [t.obs_.data for t in self.transitions]
        return torch.from_numpy(np.array(observations, dtype=np.float32)).to(self.device)

    @cached_property
    def all_extras(self):
        extras = [self.transitions[0].obs.extras] + [t.obs_.extras for t in self.transitions]
        return torch.from_numpy(np.array(extras, dtype=np.float32)).to(self.device)

    @cached_property
    def all_available_actions(self):
        available_actions = [self.transitions[0].obs.available_actions] + [t.obs_.available_actions for t in self.transitions]
        return torch.from_numpy(np.array(available_actions, dtype=np.int64)).to(self.device)

    @cached_property
    def all_states(self):
        states = [self.transitions[0].obs.state] + [t.obs_.state for t in self.transitions]
        return torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)

    @cached_property
    def actions(self):
        np_actions = np.array([t.action for t in self.transitions], dtype=np.int64)
        torch_actions = torch.from_numpy(np_actions).unsqueeze(-1).to(self.device)
        return torch_actions

    @cached_property
    def rewards(self):
        return torch.from_numpy(np.array([t.reward for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def dones(self):
        return torch.from_numpy(np.array([t.done for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def masks(self):
        return torch.ones(self.size).to(self.device)
