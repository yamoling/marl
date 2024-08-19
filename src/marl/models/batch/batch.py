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
    def all_obs_(self) -> torch.Tensor:
        """
        The first observation of the batch followed by the
        next observations of the batch.

        i.e: all observations from t=0 (reset) up to the end.
        """
        first_obs = self.obs[0].unsqueeze(0)
        return torch.cat([first_obs, self.obs_])

    @cached_property
    def all_extras_(self) -> torch.Tensor:
        """All extra information from t=0 (reset) up to the end."""
        first_extras = self.extras[0].unsqueeze(0)
        return torch.cat([first_extras, self.extras_])

    @abstractmethod  # type: ignore
    @cached_property
    def obs(self) -> torch.Tensor:
        """Observations"""

    @abstractmethod  # type: ignore
    @cached_property
    def obs_(self) -> torch.Tensor:
        """Next observations"""

    @abstractmethod  # type: ignore
    @cached_property
    def extras(self) -> torch.Tensor:
        """Extra information"""

    @abstractmethod  # type: ignore
    @cached_property
    def extras_(self) -> torch.Tensor:
        """Next extra information"""

    @abstractmethod  # type: ignore
    @cached_property
    def available_actions(self) -> torch.Tensor:
        """Available actions"""

    @abstractmethod  # type: ignore
    @cached_property
    def available_actions_(self) -> torch.Tensor:
        """Next available actions"""

    @abstractmethod  # type: ignore
    @cached_property
    def states(self) -> torch.Tensor:
        """Environment states"""

    @abstractmethod  # type: ignore
    @cached_property
    def states_(self) -> torch.Tensor:
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
