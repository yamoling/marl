from abc import ABC, abstractmethod
from marl.nn import NN
import torch

class Mixer(NN[torch.Tensor], ABC):
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

    def summary(self) -> dict[str, ]:
        return {
            "name": self.name,
            "n_agents": self.n_agents
        }