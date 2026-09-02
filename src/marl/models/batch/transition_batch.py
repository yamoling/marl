from functools import cached_property
from typing import Optional

import numpy as np
import torch
from marlenv import Transition

from .batch import Batch


class TransitionBatch(Batch):
    def __init__(self, transitions: list[Transition], device: Optional[torch.device] = None):
        self.transitions = transitions
        # self.is_continuous = np.issubdtype(transitions[0].action.dtype, np.floating)
        # self.is_discrete = not self.is_continuous
        super().__init__(len(transitions), transitions[0].n_agents, device)
        self._cache = dict[str, torch.Tensor]()

    @cached_property
    def reward_size(self):
        if self.rewards.dim() == 1:
            return 1
        return self.rewards.shape[-1]

    def multi_objective(self):
        self.actions = self.actions.unsqueeze(-1).repeat(*(1 for _ in self.actions.shape), self.reward_size)
        # This transformation is done already in the cached_prodperty of done and masks
        # self.dones = self.dones.unsqueeze(-1).repeat(*(1 for _ in self.dones.shape), self.reward_size)
        # self.masks = self.masks.unsqueeze(-1).repeat(*(1 for _ in self.masks.shape), self.reward_size)
        if self.importance_sampling_weights is not None:
            self.importance_sampling_weights = self.importance_sampling_weights.unsqueeze(-1).repeat(
                *(1 for _ in self.importance_sampling_weights.shape), self.reward_size
            )

    def __getitem__(self, key: str):
        if key in self._cache:
            return self._cache[key]
        items = np.array([t[key] for t in self.transitions])
        res = torch.from_numpy(items).to(self.device)
        self._cache[key] = res
        return res

    def get_minibatch(self, indices_or_size):
        """
        Return a minibatch built by index-selecting this batch's already materialized device tensors,
        instead of rebuilding a `TransitionBatch` from the raw `Transition` objects (which would re-run
        `np.array`/`torch.from_numpy` and a host-to-device copy for every field, every time this method is
        called).

        @ai-generated
        """
        if isinstance(indices_or_size, int):
            indices = np.random.choice(self.size, indices_or_size, replace=False)
        else:
            indices = indices_or_size
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self._index_select(index_tensor)

    def _index_select(self, index_tensor: torch.Tensor) -> "TransitionBatch":
        """
        Build a child `TransitionBatch` out of index-selections of this batch's already materialized
        tensors (both `__dict__` cached-property values and the `_cache` dict used by `__getitem__`).

        Fields that were never materialized on this (parent) batch are left untouched: the child keeps a
        sliced `transitions` list, so those fields still work lazily (computed from the small minibatch of
        transitions the first time they are accessed), they are simply not pre-indexed here.

        @ai-generated
        """
        index_list = index_tensor.tolist()
        child = TransitionBatch.__new__(TransitionBatch)
        child.transitions = [self.transitions[i] for i in index_list]
        Batch.__init__(child, len(child.transitions), self.n_agents, self.device)
        child._cache = {}
        child._individual_learners_applied = self._individual_learners_applied
        for key, value in self.__dict__.items():
            if key == "transitions":
                continue
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.size,):
                child.__dict__[key] = value[index_tensor]
        for key, value in self._cache.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.size,):
                child._cache[key] = value[index_tensor]
        return child

    def extend(self, data: list[Transition]) -> Batch:
        return TransitionBatch(self.transitions + data, self.device)

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
        return torch.from_numpy(np.array([t.next_obs.extras for t in self.transitions], dtype=np.float32)).to(
            self.device
        )

    @cached_property
    def actions(self):
        np_actions = np.array([t.action for t in self.transitions])
        torch_actions = torch.from_numpy(np_actions).to(self.device)
        return torch_actions

    @cached_property
    def rewards(self):
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        rewards = torch.from_numpy(rewards).to(self.device)
        # If the reward has only one dimension, we squeeze it
        return rewards.squeeze(-1)

    @cached_property
    def dones(self) -> torch.Tensor:
        np_dones = np.array([t.done for t in self.transitions], dtype=np.bool)
        dones = torch.from_numpy(np_dones).to(self.device)
        if self.reward_size > 1:
            dones = dones.unsqueeze(-1).expand_as(self.rewards)
        return dones

    @cached_property
    def available_actions(self):
        return torch.from_numpy(np.array([t.obs.available_actions for t in self.transitions], dtype=np.bool)).to(
            self.device
        )

    @cached_property
    def next_available_actions(self):
        return torch.from_numpy(np.array([t.next_obs.available_actions for t in self.transitions], dtype=np.bool)).to(
            self.device
        )

    @cached_property
    def states(self):
        return torch.from_numpy(np.array([t.state.data for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def states_extras(self):
        return torch.from_numpy(np.array([t.state.extras for t in self.transitions], dtype=np.float32)).to(self.device)

    @cached_property
    def next_states(self):
        return torch.from_numpy(np.array([t.next_state.data for t in self.transitions], dtype=np.float32)).to(
            self.device
        )

    @cached_property
    def next_states_extras(self):
        return torch.from_numpy(np.array([t.next_state.extras for t in self.transitions], dtype=np.float32)).to(
            self.device
        )

    @cached_property
    def masks(self):
        return torch.ones(self.size).to(self.device)

    @cached_property
    def probs(self):
        return torch.from_numpy(np.array([t.probs for t in self.transitions], dtype=np.float32)).to(self.device)  # type:ignore
