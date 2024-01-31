from abc import ABC, abstractmethod
from dataclasses import dataclass

from marl.nn import NN

import torch


@dataclass(eq=False)
class Mixer(NN[torch.Tensor], ABC):
    n_agents: int

    def __init__(self, n_agents: int):
        super().__init__((n_agents,), (0,), (1,))
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
