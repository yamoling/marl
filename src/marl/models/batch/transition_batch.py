from functools import cached_property

import numpy as np
import torch
from marlenv import Transition

from .batch import Batch


class TransitionBatch(Batch):
    def __init__(
        self,
        transitions: list[Transition],
        gamma: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        super().__init__(len(transitions), transitions[0].n_agents, gamma, device)
        self.transitions = transitions
        self._cache = dict[str, torch.Tensor]()
        n = self.size
        t0 = transitions[0]
        obs_shape = t0.obs.data.shape
        extras_shape = t0.obs.extras.shape
        first_action = np.asarray(t0.action)
        action_shape = first_action.shape
        action_dtype = first_action.dtype
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

        def to_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(self.device, non_blocking=True)

        self.obs = to_tensor(np_obs)
        self.next_obs = to_tensor(np_next_obs)
        self.extras = to_tensor(np_extras)
        self.next_extras = to_tensor(np_next_extras)
        self.actions = to_tensor(np_actions)
        self.rewards = to_tensor(np_rewards).squeeze(-1)
        self.dones = to_tensor(np_dones)
        if self.reward_size > 1:
            self.dones = self.dones.unsqueeze(-1).expand_as(self.rewards)
        self.available_actions = to_tensor(np_available_actions)
        self.next_available_actions = to_tensor(np_next_available_actions)

    @property
    def obs(self) -> torch.Tensor:
        return self._obs

    @obs.setter
    def obs(self, value: torch.Tensor) -> None:
        self._obs = value

    @property
    def next_obs(self) -> torch.Tensor:
        return self._next_obs

    @next_obs.setter
    def next_obs(self, value: torch.Tensor) -> None:
        self._next_obs = value

    @property
    def extras(self) -> torch.Tensor:
        return self._extras

    @extras.setter
    def extras(self, value: torch.Tensor) -> None:
        self._extras = value

    @property
    def next_extras(self) -> torch.Tensor:
        return self._next_extras

    @next_extras.setter
    def next_extras(self, value: torch.Tensor) -> None:
        self._next_extras = value

    @property
    def actions(self) -> torch.Tensor:
        return self._actions

    @actions.setter
    def actions(self, value: torch.Tensor) -> None:
        self._actions = value

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards

    @rewards.setter
    def rewards(self, value: torch.Tensor) -> None:
        self._rewards = value

    @property
    def dones(self) -> torch.Tensor:
        return self._dones

    @dones.setter
    def dones(self, value: torch.Tensor) -> None:
        self._dones = value

    @property
    def available_actions(self) -> torch.Tensor:
        return self._available_actions

    @available_actions.setter
    def available_actions(self, value: torch.Tensor) -> None:
        self._available_actions = value

    @property
    def next_available_actions(self) -> torch.Tensor:
        return self._next_available_actions

    @next_available_actions.setter
    def next_available_actions(self, value: torch.Tensor) -> None:
        self._next_available_actions = value

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
        items = np.array([t[key] for t in self.transitions], dtype=np.float32)
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
        Slice materialized tensors without rerunning the constructor. Uncached fields stay lazy.

        @ai-generated
        """
        index_list = index_tensor.tolist()
        child = TransitionBatch.__new__(TransitionBatch)
        child.transitions = [self.transitions[i] for i in index_list]
        Batch.__init__(child, len(child.transitions), self.n_agents, gamma=self.gamma, device=self.device)
        child._cache = {}
        child.reward_size = self.reward_size
        child._individual_learners_applied = self._individual_learners_applied
        for key, value in vars(self).items():
            if key == "gamma":
                continue
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.size,):
                setattr(child, key, value[index_tensor])
        for key, value in self._cache.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.size,):
                child._cache[key] = value[index_tensor]
        return child

    def extend(self, data: list[Transition]) -> Batch:
        return TransitionBatch(self.transitions + data, gamma=self.gamma, device=self.device)

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
        """Validity per transition and objective, before individual expansion. @ai-generated"""
        shape = (self.size, self.reward_size) if self.reward_size > 1 else (self.size,)
        return torch.ones(shape, device=self.device)

    @property
    def episode_ends(self):
        """Stop temporal traces at either termination or truncation. @ai-generated"""
        ends = torch.tensor([t.is_terminal for t in self.transitions], dtype=torch.bool, device=self.device)
        return ends.reshape(self.size, *(1 for _ in self.dones.shape[1:])).expand_as(self.dones)

    @cached_property
    def probs(self):
        return torch.from_numpy(np.array([t.probs for t in self.transitions], dtype=np.float32)).to(self.device)  # type:ignore
