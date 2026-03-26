from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
from marlenv import Episode, Transition

from .nn import NN

if TYPE_CHECKING:
    from marl.models import Batch


@dataclass
class IRModule(NN):
    """Intrinsic Reward Module: a class that adds intrinsic rewards."""

    name: str

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, batch: "Batch") -> torch.Tensor:
        """Compute the intrinsic reward for the given batch."""

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def update_episode(self, episode: Episode, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def update(self, batch: "Batch", time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def save(self, to_directory: str):
        """Save the IR Module to the given path."""
        raise NotImplementedError()

    def load(self, from_directory: str):
        """Load the IR Module from the given path."""
        raise NotImplementedError()
