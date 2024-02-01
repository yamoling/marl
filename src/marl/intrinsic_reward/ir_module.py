from abc import ABC, abstractmethod
from typing_extensions import Self
import torch
from dataclasses import dataclass
from serde import serde

from marl.models import Batch


@serde
@dataclass
class IRModule(ABC):
    """Intrinsic Reward Module."""

    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, batch: Batch) -> torch.Tensor:
        """Compute the intrinsic reward for the given batch."""

    def update(self) -> float:
        """
        Update the IR Module to train it.
        Returns the loss.
        """
        raise NotImplementedError()

    def save(self, to_directory: str):
        """Save the IR Module to the given path."""
        raise NotImplementedError()

    def load(self, from_directory: str):
        """Load the IR Module from the given path."""
        raise NotImplementedError()

    def to(self, device: torch.device) -> Self:
        """Move the IR Module to the given device."""
        raise NotImplementedError()

    def randomize(self):
        """Randomize the Intrinsic Reward Module."""
        raise NotImplementedError()
