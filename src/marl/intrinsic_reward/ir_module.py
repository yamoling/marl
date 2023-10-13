from abc import ABC, abstractmethod
from typing_extensions import Self
import torch

from marl.models import Batch

class IRModule(ABC):
    """Intrinsic Reward Module."""
    

    @abstractmethod
    def compute(self, batch: Batch) -> torch.Tensor:
        """Compute the intrinsic reward for the given batch."""

    def update(self):
        """Update the IR Module"""

    def save(self, to_directory: str):
        """Save the IR Module to the given path."""
        raise NotImplementedError()

    def load(self, from_directory: str):
        """Load the IR Module from the given path."""
        raise NotImplementedError()

    def to(self, device: torch.device) -> Self:
        """Move the IR Module to the given device."""
        raise NotImplementedError()