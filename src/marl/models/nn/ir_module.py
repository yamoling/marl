from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from marlenv import Episode, Transition

from marl.models.batch import Batch
from marl.utils.has_device import HasDevice


@dataclass
class IRModule(HasDevice):
    """Intrinsic Reward Module: a class that adds intrinsic rewards."""

    name: str

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, batch) -> torch.Tensor:
        """Compute the intrinsic reward for the given batch."""

    def update_step(self, transition: Transition, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def update_episode(self, episode: Episode, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def update(self, batch: Batch, time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}

    def save(self, to_directory: str):
        """Save the IR Module to the given path."""
        raise NotImplementedError()

    def load(self, from_directory: str):
        """Load the IR Module from the given path."""
        raise NotImplementedError()
