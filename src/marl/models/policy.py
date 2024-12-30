from abc import abstractmethod
import numpy as np
from dataclasses import dataclass


@dataclass
class Policy:
    """
    A policy takes decides which action to take given an input.
    """

    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        """
        Choose an action based on the given qvalues and avalable actions.
        Returns the chosen action.
        """

    @abstractmethod
    def update(self, time_step: int) -> dict[str, float]:
        """Update the object and return the corresponding logs."""
