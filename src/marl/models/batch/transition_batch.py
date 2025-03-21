from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Transition

from .batch import Batch


class TransitionBatch(Batch):
    def __init__(self, transitions: list[Transition[npt.NDArray]], device: Optional[torch.device] = None):
        self.transitions = transitions
        self.is_continuous = transitions[0].action.dtype in (np.float32, np.float64)
        self.is_discrete = not self.is_continuous
        self.actions_dtype = transitions[0].action.dtype
        super().__init__(len(transitions), transitions[0].n_agents, device)

    def multi_objective(self):
        self.actions = self.actions.unsqueeze(-1).repeat(*(1 for _ in self.actions.shape), self.reward_size)
        self.dones = self.dones.unsqueeze(-1).repeat(*(1 for _ in self.dones.shape), self.reward_size)
        self.masks = self.masks.unsqueeze(-1).repeat(*(1 for _ in self.masks.shape), self.reward_size)
        if self.importance_sampling_weights is not None:
            self.importance_sampling_weights = self.importance_sampling_weights.unsqueeze(-1).repeat(
                *(1 for _ in self.importance_sampling_weights.shape), self.reward_size
            )

    def __getitem__(self, key: str) -> torch.Tensor:
        items = np.array([t[key] for t in self.transitions])
        return torch.from_numpy(items).to(self.device)

    def get_minibatch(self, indices_or_size):
        if isinstance(indices_or_size, int):
            indices = np.random.choice(self.size, indices_or_size, replace=False)
        else:
            indices = indices_or_size
        return TransitionBatch([self.transitions[i] for i in indices], self.device)

    @cached_property
    def obs(self):
        return torch.from_numpy(np.array([t.obs.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def next_obs(self):
        return torch.from_numpy(np.array([t.next_obs.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def extras(self):
        return torch.from_numpy(np.array([t.obs.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def next_extras(self):
        return torch.from_numpy(np.array([t.next_obs.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def actions(self):
        np_actions = np.array([t.action for t in self.transitions], dtype=self.actions_dtype)
        torch_actions = torch.from_numpy(np_actions).to(self.device)
        return torch_actions

    @cached_property
    def rewards(self):
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        rewards = torch.from_numpy(rewards).to(self.device)
        # If the reward has only one dimension, we squeeze it
        return rewards.squeeze(-1)

    @cached_property
    def dones(self):
        dones = np.array([t.done * self.reward_size for t in self.transitions], dtype=np.float32)
        dones = torch.from_numpy(dones).to(self.device)
        if self.reward_size > 1:
            dones = dones.unsqueeze(-1).expand_as(self.rewards)
        return dones

    @cached_property
    def available_actions(self):
        return torch.from_numpy(np.array([t.obs.available_actions for t in self.transitions], dtype=np.bool)).to(self.device)

    @cached_property
    def next_available_actions(self):
        return torch.from_numpy(np.array([t.next_obs.available_actions for t in self.transitions], dtype=np.bool)).to(self.device)

    @cached_property
    def states(self):
        return torch.from_numpy(np.array([t.state.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def states_extras(self):
        return torch.from_numpy(np.array([t.state.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def next_states(self):
        return torch.from_numpy(np.array([t.next_state.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def next_states_extras(self):
        return torch.from_numpy(np.array([t.next_state.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def masks(self):
        if self.reward_size == 1:
            return torch.ones(self.size).to(self.device)
        return torch.ones(self.size, self.reward_size).to(self.device)

    @cached_property
    def probs(self):
        return torch.from_numpy(np.array([t.probs for t in self.transitions], dtype=np.float32)).to(self.device)  # type:ignore
