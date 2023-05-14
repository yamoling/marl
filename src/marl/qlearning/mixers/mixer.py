from abc import ABC, abstractmethod
from marl.utils.summarizable import Summarizable
import torch

class Mixer(Summarizable, ABC, torch.nn.Module):
    
    @abstractmethod
    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Mix the utiliy values of the agents."""

    @abstractmethod
    def save(self, to_directory: str):
        """Save the mixer to a directory."""

    @abstractmethod
    def load(self, from_directory: str):
        """Load the mixer from a directory."""
