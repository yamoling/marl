from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, Optional, overload

import torch


class Batch(ABC):
    """
    Lazy loaded batch for training.
    """

    def __init__(self, size: int, n_agents: int, device: Optional[torch.device] = None):
        super().__init__()
        self.size = size
        self.n_agents = n_agents
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.importance_sampling_weights: Optional[torch.Tensor] = None

    def for_individual_learners(self) -> "Batch":
        """Reshape rewards, dones such that each agent has its own (identical) signal."""
        if (
            self.reward_size > 1
        ):  # Need to consider this case, because multiple rewards should be at the end and dones/masks are expanded when called (so rewards needs to be as is until then)
            self.dones = self.dones.repeat_interleave(self.n_agents).view(*self.dones.shape[:-1], self.n_agents, self.dones.shape[-1])  # type:ignore
            self.masks = self.masks.repeat_interleave(self.n_agents).view(*self.masks.shape[:-1], self.n_agents, self.masks.shape[-1])
            self.rewards = self.rewards.repeat_interleave(self.n_agents).view(
                *self.rewards.shape[:-1], self.n_agents, self.rewards.shape[-1]
            )
        else:
            self.rewards = self.rewards.repeat_interleave(self.n_agents).view(*self.rewards.shape, self.n_agents)
            self.dones = self.dones.repeat_interleave(self.n_agents).view(*self.dones.shape, self.n_agents)  # type:ignore
            self.masks = self.masks.repeat_interleave(self.n_agents).view(*self.masks.shape, self.n_agents)
        return self

    def __len__(self) -> int:
        return self.size

    @abstractmethod
    def __getitem__(self, key: str) -> torch.Tensor:
        """Retrieve a dynamic attribute of the batch."""

    def normalize_rewards(self):
        """Normalize the rewards of the batch such that they have a mean of 0 and a std of 1."""
        self.rewards = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-8)

    def _normalize(self, tensor: torch.Tensor):
        """Normalize the tensor such that it has a mean of 0 and a std of 1."""
        return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

    def compute_mc_returns(self, gamma: float, next_value: torch.Tensor, normalize: bool = True):
        """
        Compute the advantages using the Monte Carlo method, i.e. the discounted sum of rewards until the end of the episode.
        """
        returns = torch.empty_like(self.rewards, dtype=torch.float32)
        for t in range(self.size - 1, -1, -1):
            next_value = self.rewards[t] + gamma * next_value * self.not_dones[t]
            returns[t] = next_value
        if normalize:
            returns = self._normalize(returns)
        return returns

    def compute_td1_returns(self, gamma: float, next_values: torch.Tensor, normalize: bool = False):
        """
        Compute the returns using the 1-step TD error.

        Args:
            next_values: Value estimate of the next states.
        """
        returns = self.rewards + gamma * next_values * self.not_dones
        if normalize:
            returns = self._normalize(returns)
        return returns

    def compute_mc_advantages(self, gamma: float, all_values: torch.Tensor, normalize: bool = False):
        """
        Compute the advantages using the Monte Carlo method, i.e. the discounted sum of advantages until the end of the episode.

        Args:
            gamma: Discount factor.
            all_values: Value estimate of the states from s_t to s_{t_max + 1}. Tensor of shape (batch_size + 1, )
            normalize: Whether to normalize the advantages.
        """
        values = all_values[:-1]
        returns = self.compute_mc_returns(gamma, all_values[-1], normalize=False)
        advantages = returns - values
        if normalize:
            advantages = self._normalize(advantages)
        return advantages

    def compute_td1_advantages(self, gamma: float, all_values: torch.Tensor, normalize: bool = False):
        """
        Compute the advantages as the 1-step TD error.

        1-step TD-errors can be considered as advantages (cf: https://arxiv.org/pdf/1506.02438 second paragraph of Section 3.).

        Args:
            gamma: Discount factor.
            all_values: Estimated state values from s_t to s_{tmax+1}. Tensor of shape (batch_size + 1,).
            normalize: Whether to normalize the advantages.
        """
        next_values = all_values[1:]
        values = all_values[:-1]
        advantages = self.compute_td1_returns(gamma, next_values, normalize=False) - values
        if normalize:
            advantages = self._normalize(advantages)
        return advantages

    def compute_gae(
        self,
        gamma: float,
        all_values: torch.Tensor,
        trace_decay: float = 0.95,
        normalize: bool = False,
    ):
        """
        Compute Generalized Advantage Estimation (GAE).
        Paper: https://arxiv.org/pdf/1506.02438

        Notes:
            This method assumes that the items are adjacent in time.
            With `trace_decay=1.0`, this method is equivalent to the Monte Carlo estimate of advantages `self.compute_mc_advantages(...)`.
            With `trace_decay=0.0`, this method is equivalent to the 1-step TD error `self.compute_td1_advantages(...)`.

        Args:
            gamma: Discount factor.
            trace_decay: Smoothing factor for GAE (lambda).
            all_values: Estimated state values from s_t to s_{tmax+1}. Tensor of shape (batch_size + 1,).
            normalize: Whether to normalize the advantages at the end of the computation.

        Returns:
            Advantage estimates (batch_size,).
        """
        values = all_values[:-1]
        next_values = all_values[1:]
        deltas = self.rewards + gamma * next_values * self.not_dones - values
        gae = torch.zeros(self.reward_size, dtype=torch.float32, device=self.device)
        advantages = torch.empty_like(self.rewards, dtype=torch.float32)
        for t in range(self.size - 1, -1, -1):
            gae = deltas[t] + gamma * trace_decay * gae
            advantages[t] = gae
        if normalize:
            advantages = self._normalize(advantages)
        return advantages

    @overload
    def get_minibatch(self, minibatch_size: int, /) -> "Batch":
        """
        Return a minibatch of the given size where the indices are randomly sampled within the range [0, self.size) without replacement.

        Args:
            minibatch_size: Size of the minibatch. Must be less than or equal to the `self.size`.
        """

    @overload
    def get_minibatch(self, indices: Iterable[int], /) -> "Batch":
        """
        Return a minibatch of the given indices.

        Args:
            indices: Indices of the minibatch. Each index must be within the range [0, self.size)
        """

    @abstractmethod
    def get_minibatch(self, arg, /) -> "Batch": ...

    @cached_property
    def n_actions(self) -> int:
        """Number of possible actions"""
        return self.available_actions.shape[-1]

    @cached_property
    def reward_size(self) -> int:
        """Number of rewards, i.e. the number of objectives"""
        if self.rewards.dim() == 1:
            return 1
        return self.rewards.shape[-1]

    @abstractmethod
    def multi_objective(self):
        """Prepare the batch for multi-objective training"""

    @cached_property
    def one_hot_actions(self) -> torch.Tensor:
        """One hot encoded actions"""
        # Actions have a last dimension of size 1 that we have to remove
        actions = self.actions.squeeze(-1)
        one_hot = torch.nn.functional.one_hot(actions, self.n_actions)
        return one_hot

    @cached_property
    def all_obs(self) -> torch.Tensor:
        """
        The first observation of the batch followed by the
        next observations of the batch.

        i.e: all observations from t=0 (reset) up to the end.
        """
        first_obs = self.obs[0].unsqueeze(0)
        return torch.cat([first_obs, self.next_obs])

    @cached_property
    def all_extras(self) -> torch.Tensor:
        """All extra information from t=0 (reset) up to the end."""
        first_extras = self.extras[0].unsqueeze(0)
        return torch.cat([first_extras, self.next_extras])

    @cached_property
    def all_available_actions(self) -> torch.Tensor:
        """All available actions from t=0 (reset) up to the end."""
        first_available_actions = self.available_actions[0].unsqueeze(0)
        return torch.cat([first_available_actions, self.next_available_actions])

    @cached_property
    def all_states(self) -> torch.Tensor:
        """All environment states from t=0 (reset) up to the end."""
        first_states = self.states[0].unsqueeze(0)
        return torch.cat([first_states, self.next_states])

    @abstractmethod  # type: ignore
    @cached_property
    def obs(self) -> torch.Tensor:
        """Observations"""

    @abstractmethod  # type: ignore
    @cached_property
    def next_obs(self) -> torch.Tensor:
        """Next observations"""

    @abstractmethod  # type: ignore
    @cached_property
    def extras(self) -> torch.Tensor:
        """Extra information"""

    @abstractmethod  # type: ignore
    @cached_property
    def next_extras(self) -> torch.Tensor:
        """Next extra information"""

    @abstractmethod  # type: ignore
    @cached_property
    def states_extras(self) -> torch.Tensor:
        """State extra information"""

    @abstractmethod  # type: ignore
    @cached_property
    def next_states_extras(self) -> torch.Tensor:
        """Next state extra information"""

    @abstractmethod  # type: ignore
    @cached_property
    def available_actions(self) -> torch.Tensor:
        """Available actions"""

    @abstractmethod  # type: ignore
    @cached_property
    def next_available_actions(self) -> torch.Tensor:
        """Next available actions"""

    @abstractmethod  # type: ignore
    @cached_property
    def states(self) -> torch.Tensor:
        """Environment states"""

    @abstractmethod  # type: ignore
    @cached_property
    def next_states(self) -> torch.Tensor:
        """Next environment states"""

    @abstractmethod  # type: ignore
    @cached_property
    def actions(self) -> torch.Tensor:
        """Actions"""

    @abstractmethod  # type: ignore
    @cached_property
    def rewards(self) -> torch.Tensor:
        """Rewards"""

    @abstractmethod  # type: ignore
    @cached_property
    def dones(self) -> torch.BoolTensor:
        """Done masks. `True` is the corresponding transition lead to a terminal state, `False` otherwise."""

    @property
    def not_dones(self) -> torch.BoolTensor:
        """Whether the corresponding transition lead to a non-terminal state. True for "continued" states, False for terminal states."""
        return ~self.dones  # type: ignore

    @abstractmethod  # type: ignore
    @cached_property
    def masks(self) -> torch.Tensor:
        """Masks (for padded episodes)"""

    @abstractmethod  # type: ignore
    @cached_property
    def probs(self) -> torch.Tensor:
        """Probabilities"""

    def to(self, device: torch.device):
        """Send the tensors to the given device"""
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                setattr(self, key, value)
        return self
