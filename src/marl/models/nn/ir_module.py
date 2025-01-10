from abc import abstractmethod
from typing import Literal
from typing_extensions import Self
import torch
from dataclasses import dataclass
from marlenv import Transition, Episode

from marl.models import Batch


@dataclass
class IRModule:
    """Intrinsic Reward Module: a class that adds intrinsic rewards."""

    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, batch: Batch) -> torch.Tensor:
        """Compute the intrinsic reward for the given batch."""

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def update_episode(self, episode: Episode, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def save(self, to_directory: str):
        """Save the IR Module to the given path."""
        raise NotImplementedError()

    def load(self, from_directory: str):
        """Load the IR Module from the given path."""
        raise NotImplementedError()

    def to(self, device: torch.device) -> Self:
        """Move the IR Module to the given device."""
        raise NotImplementedError()

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the Intrinsic Reward Module."""
        raise NotImplementedError()
