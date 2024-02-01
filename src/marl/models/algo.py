from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from dataclasses import dataclass
import torch
from rlenv.models import Observation


@dataclass
class RLAlgo(ABC):
    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    def randomize(self):
        """Randomize the algorithm parameters"""
        raise NotImplementedError("Not implemented for this algorithm")

    @abstractmethod
    def choose_action(self, observation: Observation) -> np.ndarray[np.int32, Any]:
        """Get the action to perform given the input observation"""

    # @abstractmethod
    def choose_action_extra(self, observation: Observation) -> tuple[np.ndarray, float, np.ndarray]:
        """Get the action to perform, obs value and actions probs given the input observation"""
        return self.choose_action(observation), self.value(observation), np.zeros(1)  # TODO : temporary solution not to break everything

    def new_episode(self):
        """
        Called when a new episode starts.

        This is required for recurrent algorithms, such as R-DQN, that need to reset their hidden states.
        """

    @abstractmethod
    def value(self, observation: Observation) -> float:
        """Get the value of the input observation"""

    def to(self, device: torch.device):
        """Move the algorithm to the specified device"""
        raise NotImplementedError("Not implemented for this algorithm")

    @abstractmethod
    def set_training(self):
        """
        Set the algorithm to training mode.
        This is useful for algorithms that have different behavior during training and testing,
        such as Dropout, BathNorm, or have different exploration strategies.
        """

    @abstractmethod
    def set_testing(self):
        """
        Set the algorithm to testing mode.
        This is useful for algorithms that have different behavior during training and testing,
        such as Dropout, BathNorm, or have different exploration strategies.
        """

    @abstractmethod
    def save(self, to_directory: str):
        """Save the algorithm to the specified directory"""

    @abstractmethod
    def load(self, from_directory: str):
        """Load the algorithm parameters from the specified directory"""
