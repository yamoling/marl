from collections.abc import Callable, Sequence
from functools import cached_property

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Episode

from .batch import Batch


class EpisodeBatch(Batch):
    """
    Batch of (padded) episodes, whose tensors have shape (time, batch, ...).

    Tensors are lazily built from the episodes. The tensors of current and next items (`obs` and `next_obs`,
    `states` and `next_states`, etc) are views of a single `all_*` tensor that is transferred to the device once.
    """

    def __init__(
        self,
        episodes: list[Episode],
        gamma: torch.Tensor | None = None,
        device: torch.device | None = None,
        pad_episodes: bool = True,
    ):
        """
        Args:
            episodes: The episodes of the batch. They may already be padded (e.g. by a parent batch).
            pad_episodes: Whether to pad to the longest episode. If False, the time dimension is the length of the
                episodes' stored sequences, which preserves the time dimension of a parent batch's padded episodes.

        @ai-edited
        """
        super().__init__(len(episodes), episodes[0].n_agents, gamma, device)
        self._lengths = [len(e) for e in episodes]
        if pad_episodes:
            self._max_episode_len = max(self._lengths)
        else:
            self._max_episode_len = max(len(e.actions) for e in episodes)
        self._base_episodes = episodes
        self._pad_episodes = pad_episodes

    @cached_property
    def episodes(self) -> list[Episode]:
        """The episodes, padded to the time dimension of the batch. @ai-generated"""
        if not self._pad_episodes:
            return self._base_episodes
        return [e.padded(self._max_episode_len) for e in self._base_episodes]

    @cached_property
    def reward_size(self):
        if self.rewards.dim() == 2:
            return 1
        return self.rewards.shape[-1]

    def compute_returns(self, gamma: float) -> torch.Tensor:
        result = torch.empty_like(self.rewards, dtype=torch.float32)
        next_step_returns = self.rewards[-1]
        result[-1] = next_step_returns
        for step in range(self._max_episode_len - 2, -1, -1):
            reward = self.rewards[step]
            next_step_returns = reward + gamma * next_step_returns
            result[step] = next_step_returns
        return result

    def get_minibatch2(self, arg, /) -> Batch:
        match arg:
            case int(minibatch_size):
                if minibatch_size > self.size:
                    raise ValueError(f"Minibatch size {minibatch_size} is greater than the batch size {self.size}")
                indices = np.random.choice(self.size, minibatch_size, replace=False)
            case indices:
                pass
        return EpisodeBatch([self.episodes[i] for i in indices], self.gamma, self.device)

    def get_minibatch(self, indices_or_size) -> Batch:
        match indices_or_size:
            case int(minibatch_size):
                indices = np.random.choice(self.size, minibatch_size, replace=False)
            case tuple() | list() | np.ndarray() as indices:
                pass
            case _:
                raise ValueError(f"Invalid minibatch size {indices_or_size}")
        return EpisodeBatch([self.episodes[i] for i in indices], self.gamma, self.device, pad_episodes=False)

    def extend(self, data: list[Episode]) -> Batch:
        return EpisodeBatch(self.episodes + data, self.gamma, self.device)

    def multi_objective(self):
        raise NotImplementedError()
        self.actions = self.actions.unsqueeze(-1).repeat(*(1 for _ in self.actions.shape), self.reward_size)

    def __getitem__(self, key: str) -> torch.Tensor:
        res = np.array([e[key] for e in self.episodes], dtype=np.float32)
        return torch.from_numpy(res).transpose(1, 0).to(self.device)

    @cached_property
    def probs(self):
        raise NotImplementedError()

    # Host to device transfers

    @property
    def _pinned(self) -> bool:
        return self.device.type == "cuda"

    def _host_buffer(self, shape: tuple[int, ...], dtype: np.dtype) -> tuple[torch.Tensor, np.ndarray]:
        """Uninitialised host tensor (pinned when the batch is on a CUDA device) and its numpy view. @ai-generated"""
        torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        buffer = torch.empty(shape, dtype=torch_dtype, pin_memory=self._pinned)
        return buffer, buffer.numpy()

    def _to_device(self, buffer: torch.Tensor) -> torch.Tensor:
        """Asynchronous when the buffer is pinned. @ai-generated"""
        return buffer.to(self.device, non_blocking=self._pinned)

    def _time_major(
        self,
        get_sequence: Callable[[Episode], Sequence[npt.ArrayLike]],
        n_extra_items: int,
        dtype: np.dtype | type | None = np.float32,
        pad_with_first: bool = False,
    ) -> torch.Tensor:
        """
        Gather a sequence of every episode into a padded tensor of shape (time + n_extra_items, batch, *item_shape).

        Args:
            get_sequence: Returns the sequence of an episode, which contains `len(episode) + n_extra_items` items
                (or more for already padded episodes, in which case the extra items are ignored).
            n_extra_items: 1 for `all_*` sequences, 0 for per-transition sequences.
            dtype: dtype of the tensor, or None to keep the dtype of the items.
            pad_with_first: Whether to pad with the first item of the sequence instead of zeros.

        @ai-generated
        """
        n_steps = self._max_episode_len + n_extra_items
        sequences = [get_sequence(e)[: length + n_extra_items] for e, length in zip(self._base_episodes, self._lengths)]
        first = np.asarray(sequences[0][0])
        buffer, array = self._host_buffer((self.size, n_steps, *first.shape), np.dtype(dtype or first.dtype))
        for i, sequence in enumerate(sequences):
            n_items = len(sequence)
            np.stack(sequence, out=array[i, :n_items], casting="unsafe")
            array[i, n_items:] = sequence[0] if pad_with_first else 0
        return self._to_device(buffer).transpose(0, 1)

    def _stack_to_device(self, arrays: list[np.ndarray]) -> torch.Tensor:
        """Stack float32 arrays into a tensor on `self.device`. @ai-generated"""
        buffer, array = self._host_buffer((len(arrays), *np.shape(arrays[0])), np.dtype(np.float32))
        np.stack(arrays, out=array, casting="unsafe")
        return self._to_device(buffer)

    # Padding-free representation of `all_obs` and `all_extras`

    @cached_property
    def _packed_indices(self) -> torch.Tensor:
        """
        Flat indices, in a (batch, time + 1) layout, of the items of `all_obs` that are not padding.

        Computed on the CPU so that unpacking does not require a device synchronisation.

        @ai-generated
        """
        n_steps = self._max_episode_len + 1
        indices = np.concatenate([np.arange(length + 1) + i * n_steps for i, length in enumerate(self._lengths)])
        return torch.from_numpy(indices).to(self.device)

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select the items of a (time + 1, batch, ...) tensor that are not padding. @ai-generated"""
        return tensor.transpose(0, 1).flatten(0, 1).index_select(0, self._packed_indices)

    @cached_property
    def packed_all_obs(self) -> torch.Tensor:
        """
        The non-padding items of `all_obs`, i.e. a tensor of shape (n_valid, *obs_shape) ordered by episode, then
        by time step. Use `unpack_all` to restore the (time + 1, batch, ...) layout of the outputs computed from it.

        Building it is much cheaper than `all_obs` when episodes have different lengths. It is selected from
        `all_obs` if the latter is already loaded, and otherwise built from the episodes (hence ignoring any
        value assigned to `obs` or `next_obs`).

        @ai-generated
        """
        if "all_obs" in self.__dict__:
            return self._pack(self.all_obs)
        return self._stack_to_device([o for e, length in zip(self._base_episodes, self._lengths) for o in e.all_observations[: length + 1]])

    @cached_property
    def packed_all_extras(self) -> torch.Tensor:
        """The non-padding items of `all_extras`, ordered like `packed_all_obs`. @ai-generated"""
        if "all_extras" in self.__dict__:
            return self._pack(self.all_extras)
        return self._stack_to_device([x for e, length in zip(self._base_episodes, self._lengths) for x in e.all_extras[: length + 1]])

    def unpack_all(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Scatter a tensor of shape (n_valid, *rest) computed from `packed_all_obs` into a zero-padded tensor of shape
        (time + 1, batch, *rest), i.e. the layout of `all_obs`.

        @ai-generated
        """
        unpacked = packed.new_zeros(self.size * (self._max_episode_len + 1), *packed.shape[1:])
        unpacked = unpacked.index_copy(0, self._packed_indices, packed)
        return unpacked.view(self.size, self._max_episode_len + 1, *packed.shape[1:]).transpose(0, 1)

    # Observations

    @cached_property
    def all_obs(self):
        """Observations from t=0 (reset) up to the end, padded with zeros. @ai-edited"""
        if "packed_all_obs" in self.__dict__:
            return self.unpack_all(self.packed_all_obs)
        return self._time_major(lambda e: e.all_observations, 1)

    @property
    def obs(self) -> torch.Tensor:
        return self._obs

    @obs.setter
    def obs(self, value: torch.Tensor) -> None:
        self._obs = value

    @cached_property
    def _obs(self) -> torch.Tensor:
        return self.all_obs[:-1]

    @property
    def next_obs(self) -> torch.Tensor:
        return self._next_obs

    @next_obs.setter
    def next_obs(self, value: torch.Tensor) -> None:
        self._next_obs = value

    @cached_property
    def _next_obs(self) -> torch.Tensor:
        return self.all_obs[1:]

    # Extras

    @cached_property
    def all_extras(self):
        """Extras from t=0 (reset) up to the end, padded with zeros. @ai-edited"""
        if "packed_all_extras" in self.__dict__:
            return self.unpack_all(self.packed_all_extras)
        return self._time_major(lambda e: e.all_extras, 1)

    @property
    def extras(self) -> torch.Tensor:
        return self._extras

    @extras.setter
    def extras(self, value: torch.Tensor) -> None:
        self._extras = value

    @cached_property
    def _extras(self) -> torch.Tensor:
        return self.all_extras[:-1]

    @property
    def next_extras(self) -> torch.Tensor:
        return self._next_extras

    @next_extras.setter
    def next_extras(self, value: torch.Tensor) -> None:
        self._next_extras = value

    @cached_property
    def _next_extras(self) -> torch.Tensor:
        return self.all_extras[1:]

    # States

    @cached_property
    def all_states(self) -> torch.Tensor:
        """States from t=0 (reset) up to the end, padded with zeros. @ai-generated"""
        return self._time_major(lambda e: e.all_states, 1)

    @cached_property
    def states(self):
        return self.all_states[:-1]

    @cached_property
    def next_states(self):
        return self.all_states[1:]

    @cached_property
    def all_states_extras(self) -> torch.Tensor:
        """State extras from t=0 (reset) up to the end, padded with zeros. @ai-generated"""
        return self._time_major(lambda e: e.all_states_extras, 1)

    @cached_property
    def states_extras(self):
        return self.all_states_extras[:-1]

    @cached_property
    def next_states_extras(self):
        return self.all_states_extras[1:]

    # Available actions

    @cached_property
    def all_available_actions(self) -> torch.Tensor:
        """Available actions from t=0 (reset) up to the end, padded with the first ones. @ai-generated"""
        return self._time_major(lambda e: e.all_available_actions, 1, np.bool, pad_with_first=True)

    @property
    def available_actions(self) -> torch.Tensor:
        return self._available_actions

    @available_actions.setter
    def available_actions(self, value: torch.Tensor) -> None:
        self._available_actions = value

    @cached_property
    def _available_actions(self) -> torch.Tensor:
        return self.all_available_actions[:-1]

    @property
    def next_available_actions(self) -> torch.Tensor:
        return self._next_available_actions

    @next_available_actions.setter
    def next_available_actions(self, value: torch.Tensor) -> None:
        self._next_available_actions = value

    @cached_property
    def _next_available_actions(self) -> torch.Tensor:
        return self.all_available_actions[1:]

    # Actions, rewards, dones and masks

    @property
    def actions(self) -> torch.Tensor:
        return self._actions

    @actions.setter
    def actions(self, value: torch.Tensor) -> None:
        self._actions = value

    @cached_property
    def _actions(self) -> torch.Tensor:
        return self._time_major(lambda e: e.actions, 0, dtype=None)

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards

    @rewards.setter
    def rewards(self, value: torch.Tensor) -> None:
        self._rewards = value

    @cached_property
    def _rewards(self) -> torch.Tensor:
        return self._time_major(lambda e: e.rewards, 0).squeeze(-1)

    @property
    def dones(self) -> torch.Tensor:
        return self._dones

    @dones.setter
    def dones(self, value: torch.Tensor) -> None:
        self._dones = value

    def _per_step_flags(self, value: Callable[[Episode, int], tuple[slice, float] | None], dtype: np.dtype | type) -> torch.Tensor:
        """
        Build a (time, batch, *reward_shape) tensor that is 0 everywhere except, for each episode, in the slice
        of time steps returned by `value(episode, length)` (or nowhere if it returns None).

        @ai-generated
        """
        reward_shape = np.shape(self._base_episodes[0].rewards[0])
        buffer, array = self._host_buffer((self.size, self._max_episode_len, *reward_shape), np.dtype(dtype))
        array[:] = 0
        for i, (episode, length) in enumerate(zip(self._base_episodes, self._lengths)):
            selected = value(episode, length)
            if selected is not None:
                array[i, selected[0]] = selected[1]
        return self._to_device(buffer).squeeze(-1).transpose(0, 1)

    @cached_property
    def _dones(self) -> torch.Tensor:
        """True from the last step of the done episodes onwards, padding included. @ai-edited"""
        return self._per_step_flags(lambda e, length: (slice(length - 1, None), True) if e.is_done else None, np.bool)

    @cached_property
    def masks(self):
        """1 on the time steps of the episodes, 0 on padding. @ai-edited"""
        return self._per_step_flags(lambda e, length: (slice(None, length), 1.0), np.float32)

    @property
    def episode_ends(self):
        """Mark the final valid step of each trajectory, including time limits. @ai-generated"""
        next_masks = torch.cat((self.masks[1:], torch.zeros_like(self.masks[:1])))
        return self.dones | (next_masks == 0)
