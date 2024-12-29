from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

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

    def compute_returns(self, gamma: float, next_value: torch.Tensor):
        returns = torch.empty_like(self.rewards)
        next_return = next_value
        for i in range(len(self.rewards), -1, -1):
            return_t = self.rewards[i] + gamma * next_return
            returns[i] = return_t
            next_return = return_t
        return returns

    def normalize_rewards(self):
        """Normalize the rewards of the batch such that they have a mean of 0 and a std of 1."""
        self.rewards = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-8)

    def compute_gae(self, values: torch.Tensor, last_next_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute the Generalized Advantage Estimation (GAE).

        Args:
            values: The values predicted by the critic
            last_next_value: The value of state after the last one (zero if the episode is done)
            gamma: The discount factor
            gae_lambda: The GAE $\\lambda$ hyperparameter

        Paper: https://arxiv.org/pdf/1506.02438
        """
        next_values = torch.cat([values[1:], last_next_value.unsqueeze(0)])
        deltas = self.rewards + gamma * (1 - self.dones) * next_values - values

        advantages = torch.empty_like(self.rewards)
        gae = torch.zeros_like(last_next_value)
        # Iterate backward through rewards to compute GAE
        for t in range(self.size - 1, -1, -1):
            # TD-error
            # delta = self.rewards[t] + gamma * (1 - self.dones[t]) * next_value - values[t]
            gae = deltas[t] + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
        return advantages

    @abstractmethod
    def get_minibatch(self, minibatch_size: int) -> "Batch":
        """Get a random minibatch from the batch."""

    @cached_property
    def n_actions(self) -> int:
        """Number of possible actions"""
        return self.available_actions.shape[-1]

    @cached_property
    def reward_size(self) -> int:
        """Number of rewards, i.e. the number of objectives"""
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
    def all_next_obs(self) -> torch.Tensor:
        """
        The first observation of the batch followed by the
        next observations of the batch.

        i.e: all observations from t=0 (reset) up to the end.
        """
        first_obs = self.obs[0].unsqueeze(0)
        return torch.cat([first_obs, self.next_obs])

    @cached_property
    def all_next_extras(self) -> torch.Tensor:
        """All extra information from t=0 (reset) up to the end."""
        first_extras = self.extras[0].unsqueeze(0)
        return torch.cat([first_extras, self.next_extras])

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
