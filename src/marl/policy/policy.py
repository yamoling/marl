from abc import abstractmethod
import numpy as np


class Policy:
    """
    A policy takes decides which action to take given an input.
    """

    @abstractmethod
    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        """
        Choose an action based on the given qvalues and avalable actions.
        Returns the chosen action.
        """

    def update(self):
        """Update the policy"""
