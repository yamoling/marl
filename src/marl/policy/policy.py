from abc import abstractmethod
import os
import json
import numpy as np
from marl.utils import Serializable


class Policy(Serializable):
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

    def save(self, to_file: str):
        """Save the policy to a directory"""
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        with open(to_file, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f)

    @classmethod
    def load(cls, from_file: str):
        """Load the policy from a directory"""
        with open(from_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            return Serializable.from_dict(summary)
    