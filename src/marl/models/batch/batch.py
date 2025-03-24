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
        self.rewards = self.rewards.repeat_interleave(self.n_agents).view(*self.rewards.shape, self.n_agents)
        self.dones = self.dones.repeat_interleave(self.n_agents).view(*self.dones.shape, self.n_agents)
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

    def compute_advantages2(self, values: torch.Tensor, next_values: torch.Tensor, gamma: float, gae_lambda: float):
        advantage = torch.zeros(self.size, dtype=torch.float32)
        for t in range(self.size - 1):
            discount = 1
            a_t = 0
            for k in range(t, self.size - 1):
                a_t += discount * (self.rewards[k] + gamma * values[k + 1] * (1 - int(self.dones[k])) - values[k])
                discount *= gamma * gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(self.device)
        return advantage

    def compute_returns(self, gamma: float, next_value: torch.Tensor, normalize: bool = True):
        returns = []
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                next_value = next_value.zero_()
            next_value = reward + next_value * gamma
            returns.insert(0, next_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def calculate_advantages(self, values, discount_factor, trace_decay, normalize=True):
        advantages = []
        advantage = 0
        next_value = 0
        for r, v in zip(reversed(self.rewards), reversed(values)):
            td_error = r + next_value * discount_factor - v
            advantage = td_error + advantage * discount_factor * trace_decay
            next_value = v
            advantages.insert(0, advantage)
        advantages = torch.tensor(advantages)
        if normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages

    def compute_advantage(self, values: torch.Tensor, next_values: torch.Tensor, gamma: float, trace_decay: float = 1.0, normalize=True):
        """
        Compute the advantages (GAE if `trace_decay` is provided).
        GAE Paper: https://arxiv.org/pdf/1506.02438
        """
        advantages = torch.empty(self.size, dtype=torch.float32)
        td_errors = self.rewards + gamma * next_values - values
        gae = 0.0
        for t in range(self.size - 1, -1, -1):
            gae = td_errors[t] + gae * gamma * trace_decay
            advantages[t] = gae
        if normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages

    @overload
    def get_minibatch(self, minibatch_size: int) -> "Batch": ...

    @overload
    def get_minibatch(self, indices: Iterable[int]) -> "Batch": ...

    @abstractmethod
    def get_minibatch(self, arg) -> "Batch":  # type: ignore
        ...

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
    def dones(self) -> torch.Tensor:
        """Dones"""

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
