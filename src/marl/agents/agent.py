import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import torch
from marlenv.models import Observation, ActionSpace

from marl.models.nn import NN, RecurrentNN


@dataclass
class Agent[A](ABC):
    name: str

    def __init__(self):
        self.name = self.__class__.__name__
        self.device = torch.device("cpu")

    @property
    def action_space(self) -> ActionSpace:
        raise NotImplementedError("Action space not implemented")

    @cached_property
    def networks(self):
        """Dynamic list of neural networks attributes in the agent"""
        return [nn for nn in self.__dict__.values() if isinstance(nn, NN)]

    @cached_property
    def recurrent_networks(self):
        """Dynamic list of recurrent neural networks attributes in the agent"""
        return [nn for nn in self.networks if isinstance(nn, RecurrentNN)]

    def seed(self, seed: int):
        """
        Seed the algorithm for reproducibility (e.g. during testing).

        Seed `ranom`, `numpy`, and `torch` libraries by default.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the algorithm parameters"""
        for nn in self.networks:
            nn.randomize(method)

    @abstractmethod
    def get_action_distribution(self, observation: Observation):
        """Get an action distribution given the input observation."""
        raise NotImplementedError("Getting action distribution not implemented")

    @abstractmethod
    def choose_action(self, observation: Observation) -> A:
        """Get the action to perform given the input observation"""

    def new_episode(self):
        """
        Called when a new episode starts.

        This is required for recurrent algorithms such as R-DQN that need to reset their hidden states.
        """
        for nn in self.recurrent_networks:
            nn.reset_hidden_states()

    def value(self, obs: Observation) -> float:
        """Get the value of the input observation"""
        return 0.0

    def to(self, device: torch.device):
        """Move the algorithm to the specified device"""
        self.device = device
        for nn in self.networks:
            nn.to(device)
        return self

    def set_training(self):
        """
        Set the algorithm to training mode.
        This is useful for algorithms that have different behavior during training and testing,
        such as Dropout, BathNorm, or have different exploration strategies.
        """
        for nn in self.networks:
            nn.train()

    def set_testing(self):
        """
        Set the algorithm to testing mode.
        This is useful for algorithms that have different behavior during training and testing,
        such as Dropout, BathNorm, or have different exploration strategies.
        """
        for nn in self.networks:
            nn.eval()

    @property
    def __can_autosave(self):
        """Check if the algorithm can be autosaved"""
        names = set(nn.name for nn in self.networks)
        return len(names) == len(self.networks)

    def save(self, to_directory: str):
        """Save the algorithm to the specified directory"""
        if not self.__can_autosave:
            raise NotImplementedError("Duplicate network name, you need to implement a custom save method")
        os.makedirs(to_directory, exist_ok=True)
        for nn in self.networks:
            torch.save(nn.state_dict(), os.path.join(to_directory, f"{nn.name}.pt"))

    def load(self, from_directory: str):
        """Load the algorithm parameters from the specified directory"""
        if not self.__can_autosave:
            raise NotImplementedError("Duplicate network name, you need to implement a custom load method")
        for nn in self.networks:
            nn.load_state_dict(torch.load(os.path.join(from_directory, f"{nn.name}.pt")))
