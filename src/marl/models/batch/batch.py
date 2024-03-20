from abc import ABC, abstractmethod
from typing import Optional
from functools import cached_property
import torch


class Batch(ABC):
    """
    Lazy loaded batch for training.
    """

    def __init__(self, size: int, n_agents: int):
        super().__init__()
        self.size = size
        self.n_agents = n_agents
        self.device = torch.device("cpu")
        self.importance_sampling_weights: Optional[torch.Tensor] = None

    def for_individual_learners(self) -> "Batch":
        """Reshape rewards, dones such that each agent has its own (identical) signal."""
        self.rewards = self.rewards.repeat_interleave(self.n_agents).view(*self.rewards.shape, self.n_agents)
        self.dones = self.dones.repeat_interleave(self.n_agents).view(*self.dones.shape, self.n_agents)
        return self

    def __len__(self) -> int:
        return self.size

    @cached_property
    def n_actions(self) -> int:
        """Number of possible actions"""
        return self.available_actions.shape[-1]

    @cached_property
    def reward_size(self) -> int:
        """Shape of the reward tensor"""
        return self.rewards.shape[-1]

    def multi_objective(self):
        self.actions = self.actions.unsqueeze(-1).repeat(*(1 for _ in self.actions.shape), self.reward_size)
        self.dones = self.dones.unsqueeze(-1).repeat(*(1 for _ in self.dones.shape), self.reward_size)
        self.masks = self.masks.unsqueeze(-1).repeat(*(1 for _ in self.masks.shape), self.reward_size)

    @cached_property
    def obs(self) -> torch.Tensor:
        """Observations"""
        return self.all_obs[:-1]

    @cached_property
    def obs_(self) -> torch.Tensor:
        """Next observations"""
        return self.all_obs[1:]

    @cached_property
    def extras(self) -> torch.Tensor:
        """Extra information"""
        return self.all_extras[:-1]

    @cached_property
    def extras_(self) -> torch.Tensor:
        """Nest extra information"""
        return self.all_extras[1:]

    @cached_property
    def one_hot_actions(self) -> torch.Tensor:
        """One hot encoded actions"""
        # Actions have a last dimension of size 1 that we have to remove
        actions = self.actions.squeeze(-1)
        one_hot = torch.nn.functional.one_hot(actions, self.n_actions)
        return one_hot

    @cached_property
    def available_actions(self) -> torch.Tensor:
        """Available actions"""
        return self.all_available_actions[:-1]

    @cached_property
    def available_actions_(self) -> torch.Tensor:
        """Next available actions"""
        return self.all_available_actions[1:]

    @cached_property
    def states(self) -> torch.Tensor:
        """Environment states"""
        return self.all_states[:-1]

    @cached_property
    def states_(self) -> torch.Tensor:
        """Next environment states"""
        return self.all_states[1:]

    @abstractmethod  # type: ignore
    @cached_property
    def all_obs(self) -> torch.Tensor:
        """All observations from t=0 up to episode_length + 1."""

    @abstractmethod  # type: ignore
    @cached_property
    def all_extras(self) -> torch.Tensor:
        """All extra information from t=0 up to episode_length + 1."""

    @abstractmethod  # type: ignore
    @cached_property
    def all_available_actions(self) -> torch.Tensor:
        """All available actions from t=0 up to episode_length + 1."""

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
    def all_states(self) -> torch.Tensor:
        """Environment states"""

    @abstractmethod  # type: ignore
    @cached_property
    def masks(self) -> torch.Tensor:
        """Masks (for padded episodes)"""

    def to(self, device: torch.device):
        """Send the tensors to the given device"""
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                setattr(self, key, value)
        return self
