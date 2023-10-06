from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
import torch

@dataclass
class Batch(ABC):
    """
    Lazy loaded batch for training.
    Every field is set to None by default. When the field is accessed, the specific attribute is 
    loaded with the `_get_<attribute>()` method that child classes must implement.
    (exception for importance sampling weights that are set by the memory)
    """
    def __init__(self, size: int, n_agents: int, sample_indices: list[int]) -> None:
        super().__init__()
        self.size = size
        self.n_agents = n_agents
        self.sample_indices = sample_indices
        self.device = torch.device("cpu")
        self.importance_sampling_weights = None

    def for_individual_learners(self) -> "Batch":
        """Reshape rewards, dones such that each agent has its own (identical) signal."""
        self.rewards = self.rewards.repeat_interleave(self.n_agents).view(*self.rewards.shape, self.n_agents)
        self.dones = self.dones.repeat_interleave(self.n_agents).view(*self.dones.shape, self.n_agents)
        return self

    def __len__(self) -> int:
        return self.size
    
    @abstractmethod
    @cached_property
    def obs(self) -> torch.Tensor:
        """Observations"""

    @abstractmethod
    @cached_property
    def obs_(self) -> torch.Tensor:
        """Next observations"""

    @abstractmethod
    @cached_property
    def extras(self) -> torch.Tensor:
        """Extra information"""

    @abstractmethod
    @cached_property
    def extras_(self) -> torch.Tensor:
        """Nest extra information"""

    @abstractmethod
    @cached_property
    def actions(self) -> torch.LongTensor:
        """Actions"""

    @abstractmethod
    @cached_property
    def rewards(self) -> torch.Tensor:
        """Rewards"""

    @abstractmethod
    @cached_property
    def dones(self) -> torch.Tensor:
        """Dones"""

    @abstractmethod
    @cached_property
    def available_actions_(self) -> torch.LongTensor:
        """Next available actions"""

    @abstractmethod
    @cached_property
    def available_actions(self) -> torch.LongTensor:
        """Available actions"""

    @abstractmethod
    @cached_property
    def states(self) -> torch.Tensor:
        """Environment states"""

    @abstractmethod
    @cached_property
    def states_(self) -> torch.Tensor:
        """Next environment states"""

    @abstractmethod
    @cached_property
    def action_probs(self) -> torch.Tensor:
        """Probabilities of the taken action"""
    
    def to(self, device: torch.device) -> "Batch":
        """Send the tensors to the given device"""
        self.device = device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
                setattr(self, key, value)
        return self
