from abc import abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Policy:
    """
    A policy takes decides which action to take given an input.
    """

    name: str

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray | None = None) -> np.ndarray:
        """
        Choose an action based on the given qvalues and avalable actions.
        Returns the chosen action.
        """

    @abstractmethod
    def update(self, time_step: int) -> dict[str, float]:
        """Update the object and return the corresponding logs."""
