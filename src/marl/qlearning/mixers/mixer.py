from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Self, Any

from marl.nn import NN

import torch

@dataclass(eq=False)
class Mixer(NN[torch.Tensor], ABC):
    n_agents: int

    def __init__(self, n_agents: int) -> None:
        super().__init__((n_agents, ), (0, ), (1, ))
        self.n_agents = n_agents
    
    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Mix the utiliy values of the agents."""

    @abstractmethod
    def save(self, to_directory: str):
        """Save the mixer to a directory."""

    @abstractmethod
    def load(self, from_directory: str):
        """Load the mixer from a directory."""

    @classmethod
    def from_dict(cls, summary: dict[str, Any]) -> Self:
        summary.pop("input_shape", None)
        summary.pop("output_shape", None)
        summary.pop("extras_shape", None)
        return super().from_dict(summary)