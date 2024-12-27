from functools import cached_property
from typing import Optional, Iterable, overload

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

    @overload
    def get_minibatch(self, minibatch_size: int) -> Batch: ...

    @overload
    def get_minibatch(self, indices: Iterable[int]) -> Batch: ...

    def get_minibatch(self, indices_or_size) -> Batch:  # type: ignore
        if isinstance(indices_or_size, int):
            indices = np.random.choice(self.size, indices_or_size)
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
        if self.is_discrete:
            torch_actions = torch_actions.unsqueeze(-1)
            torch_actions = torch_actions.unsqueeze(-1).repeat(*(1 for _ in torch_actions.shape), self.reward_size)
        return torch_actions

    @cached_property
    def rewards(self):
        return torch.from_numpy(np.array([t.reward for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def dones(self):
        return torch.from_numpy(np.array([[t.done] * self.reward_size for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def available_actions(self):
        return torch.from_numpy(np.array([t.obs.available_actions for t in self.transitions], dtype=np.int64)).to(self.device)

    @cached_property
    def next_available_actions(self):
        return torch.from_numpy(np.array([t.next_obs.available_actions for t in self.transitions], dtype=np.int64)).to(self.device)

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
        return torch.ones(self.size, self.reward_size).to(self.device)

    @cached_property
    def probs(self):
        return torch.from_numpy(np.array([t.probs for t in self.transitions], dtype=np.float32)).to(self.device)  # type:ignore
