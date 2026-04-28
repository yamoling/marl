import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from marlenv import Observation

from ..action import Action
from ..nn import NN, RecurrentNN


@dataclass
class Agent[T](ABC):
    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__
        self._device = torch.device("cpu")
        self._training = True

    @abstractmethod
    def choose_action(self, observation: Observation, *, with_details: bool = False) -> Action[T]:
        """
        Get the action to perform given the input observation.

        If the `with_details` flag is set to True, the method should return an Action with additional details about the decision-making such as the action probabilities, the q-values, etc.
        """

    @property
    def is_training(self):
        return self._training

    @property
    def is_testing(self):
        return not self._training

    def networks(self):
        """Dynamic list of neural networks attributes in the agent"""
        return [nn for nn in self.__dict__.values() if isinstance(nn, NN)]

    @property
    def device(self):
        """Device on which the agent is located"""
        return self._device

    @cached_property
    def recurrent_networks(self):
        """Dynamic list of recurrent neural networks attributes in the agent"""
        return [nn for nn in self.networks() if isinstance(nn, RecurrentNN)]

    def seed(self, seed: int):
        """
        Seed the algorithm for reproducibility (e.g. during testing).

        Seed `random`, `numpy`, and `torch` libraries by default.
        """
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the algorithm parameters"""
        for nn in self.networks():
            nn.randomize(method)

    def new_episode(self):
        """
        Called when a new episode starts.

        This is required for recurrent algorithms such as R-DQN that need to reset their hidden states.
        """
        for nn in self.recurrent_networks:
            nn.reset_hidden_states()

    def to(self, device: torch.device):
        """Move the algorithm to the specified device"""
        self._device = device
        for nn in self.networks():
            nn.to(device)
        return self

    def set_training(self):
        """
        Set the algorithm to training mode.
        This is useful for algorithms that have different behavior during training and testing,
        such as Dropout, BathNorm, or have different exploration strategies.
        """
        self._training = True
        for nn in self.networks():
            nn.train()

    def set_testing(self):
        """
        Set the algorithm to testing mode.
        This is useful for algorithms that have different behavior during training and testing,
        such as Dropout, BathNorm, or have different exploration strategies.
        """
        self._training = False
        for nn in self.networks():
            nn.eval()

    def _can_autosave(self):
        """Check if the algorithm can be autosaved"""
        networks = self.networks()
        names = set(nn.name for nn in networks)
        return len(names) == len(networks)

    def save(self, to_directory: str):
        """Save the algorithm to the specified directory"""
        if not self._can_autosave():
            raise NotImplementedError("Duplicate network name, you need to implement a custom save method")
        os.makedirs(to_directory, exist_ok=True)
        for nn in self.networks():
            torch.save(nn.state_dict(), os.path.join(to_directory, f"{nn.name}.pt"))

    def load(self, from_directory: str):
        """Load the algorithm parameters from the specified directory"""
        if not self._can_autosave():
            raise NotImplementedError("Duplicate network name, you need to implement a custom load method")
        for nn in self.networks():
            nn.load_state_dict(torch.load(os.path.join(from_directory, f"{nn.name}.pt"), map_location=self.device))
