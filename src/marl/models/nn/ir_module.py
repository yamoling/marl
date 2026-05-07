from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .nn import NN

if TYPE_CHECKING:
    from marl.models import Batch


@dataclass
class IRModule(NN):
    """Intrinsic Reward Module: a class that adds intrinsic rewards."""

    @abstractmethod
    def compute(self, batch: "Batch") -> torch.Tensor:
        """Compute the intrinsic reward for the given batch."""

    def update(self, batch: "Batch", time_step: int) -> dict[str, float]:
        """Update the Intrinsic Reward Module."""
        return {}
