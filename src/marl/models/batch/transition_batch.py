from functools import cached_property
from typing import Optional

import numpy as np
import torch
from marlenv import Transition

from .batch import Batch

# Fields that are packed in a single pass over the transitions in `_pack` (see there). Keep this in
# sync with the fields filled by `_pack`.
_PACKED_FIELDS = (
    "obs",
    "next_obs",
    "extras",
    "next_extras",
    "actions",
    "rewards",
    "dones",
    "available_actions",
    "next_available_actions",
)

# Pinning a freshly allocated host tensor before the H->D copy has its own cost. Benchmarked against
# the unpinned variant (see reports/optimizations/03-transition-batch-single-pass.md); keep whichever
# is faster.
_PIN_MEMORY = False


class TransitionBatch(Batch):
    def __init__(self, transitions: list[Transition], device: Optional[torch.device] = None):
        self.transitions = transitions
        # self.is_continuous = np.issubdtype(transitions[0].action.dtype, np.floating)
        # self.is_discrete = not self.is_continuous
        super().__init__(len(transitions), transitions[0].n_agents, device)
        self._cache = dict[str, torch.Tensor]()
        self._packed = False

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
        # If every single-pass-packed field was already materialized on the parent (and therefore
        # copied above), the child is already fully packed and must not repack from its (sliced)
        # transitions. Otherwise, leave it lazy: missing fields will be packed on first access.
        child._packed = all(name in child.__dict__ for name in _PACKED_FIELDS)
        return child

    def extend(self, data: list[Transition]) -> Batch:
        return TransitionBatch(self.transitions + data, self.device)

    def _pack(self):
        """
        Materialize the fields that are always needed by trainers (`obs`, `next_obs`, `extras`,
        `next_extras`, `actions`, `rewards`, `dones`, `available_actions`, `next_available_actions`)
        in a single pass over `self.transitions`, instead of one independent `np.array(...)` list
        comprehension per field.

        Pre-allocated NumPy arrays are filled in one `for` loop, then each field is transferred to
        `self.device` with a single transfer. When the target device is CUDA, the host tensor is
        pinned first (if `_PIN_MEMORY`) and copied asynchronously (`non_blocking=True`).

        This is triggered lazily, either by `to()` (once the final device is known) or by the first
        access of any of the packed fields (through their `cached_property` getters below), and stores
        results directly in `self.__dict__` so that the corresponding `cached_property` never runs its
        own body.

        @ai-generated
        """
        if self._packed:
            return
        self._packed = True

        transitions = self.transitions
        n = self.size
        t0 = transitions[0]
        obs_shape = t0.obs.data.shape
        extras_shape = t0.obs.extras.shape
        action_shape = t0.action.shape
        action_dtype = t0.action.dtype
        reward_shape = t0.reward.shape
        avail_shape = t0.obs.available_actions.shape

        np_obs = np.empty((n, *obs_shape), dtype=np.float32)
        np_next_obs = np.empty((n, *obs_shape), dtype=np.float32)
        np_extras = np.empty((n, *extras_shape), dtype=np.float32)
        np_next_extras = np.empty((n, *extras_shape), dtype=np.float32)
        np_actions = np.empty((n, *action_shape), dtype=action_dtype)
        np_rewards = np.empty((n, *reward_shape), dtype=np.float32)
        np_dones = np.empty((n,), dtype=bool)
        np_available_actions = np.empty((n, *avail_shape), dtype=bool)
        np_next_available_actions = np.empty((n, *avail_shape), dtype=bool)

        for i, t in enumerate(transitions):
            np_obs[i] = t.obs.data
            np_next_obs[i] = t.next_obs.data
            np_extras[i] = t.obs.extras
            np_next_extras[i] = t.next_obs.extras
            np_actions[i] = t.action
            np_rewards[i] = t.reward
            np_dones[i] = t.done
            np_available_actions[i] = t.obs.available_actions
            np_next_available_actions[i] = t.next_obs.available_actions

        device = self.device
        use_cuda = device.type == "cuda"

        def to_tensor(array: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(array)
            if use_cuda:
                if _PIN_MEMORY:
                    tensor = tensor.pin_memory()
                tensor = tensor.to(device, non_blocking=True)
            elif device != tensor.device:
                tensor = tensor.to(device)
            return tensor

        self.__dict__["obs"] = to_tensor(np_obs)
        self.__dict__["next_obs"] = to_tensor(np_next_obs)
        self.__dict__["extras"] = to_tensor(np_extras)
        self.__dict__["next_extras"] = to_tensor(np_next_extras)
        self.__dict__["actions"] = to_tensor(np_actions)
        # If the reward has only one dimension, we squeeze it
        self.__dict__["rewards"] = to_tensor(np_rewards).squeeze(-1)
        dones = to_tensor(np_dones)
        if self.reward_size > 1:
            dones = dones.unsqueeze(-1).expand_as(self.rewards)
        self.__dict__["dones"] = dones
        self.__dict__["available_actions"] = to_tensor(np_available_actions)
        self.__dict__["next_available_actions"] = to_tensor(np_next_available_actions)

    @cached_property
    def obs(self):
        self._pack()
        return self.__dict__["obs"]

    @cached_property
    def next_obs(self):
        self._pack()
        return self.__dict__["next_obs"]

    @cached_property
    def extras(self):
        self._pack()
        return self.__dict__["extras"]

    @cached_property
    def next_extras(self):
        self._pack()
        return self.__dict__["next_extras"]

    @cached_property
    def actions(self):
        self._pack()
        return self.__dict__["actions"]

    @cached_property
    def rewards(self):
        self._pack()
        return self.__dict__["rewards"]

    @cached_property
    def dones(self) -> torch.Tensor:
        self._pack()
        return self.__dict__["dones"]

    @cached_property
    def available_actions(self):
        self._pack()
        return self.__dict__["available_actions"]

    @cached_property
    def next_available_actions(self):
        self._pack()
        return self.__dict__["next_available_actions"]

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
        return torch.ones(self.size, device=self.device)

    @cached_property
    def probs(self):
        return torch.from_numpy(np.array([t.probs for t in self.transitions], dtype=np.float32)).to(self.device)  # type:ignore
