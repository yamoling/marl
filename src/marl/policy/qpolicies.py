import json
import os
import random
import numpy as np

from .policy import Policy

class SoftmaxPolicy(Policy):
    """Softmax policy"""

    def __init__(self, actions: list, tau: float = 1.):
        self._actions = actions
        self._tau = tau
        """Temperature parameter"""

    def get_action(self, qvalues: np.ndarray[np.float32], available_actions: np.ndarray[np.float32]) -> np.ndarray[np.int64]:
        qvalues[available_actions == 0.] = -np.inf
        exp = np.exp(qvalues / self._tau)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        chosen_actions = [np.random.choice(self._actions, p=agent_probs) for agent_probs in probs]
        return np.array(chosen_actions)

    def update(self):
        pass

class EpsilonGreedy(Policy):
    """Epsilon Greedy policy"""

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        qvalues[available_actions == 0.] = -np.inf
        chosen_actions = qvalues.argmax(axis=-1)
        replacements = np.array([random.choice(np.nonzero(available)[0]) for available in available_actions])
        r = np.random.random(len(qvalues))
        mask = r < self._epsilon
        chosen_actions[mask] = replacements[mask]
        return chosen_actions

    def save(self, to_directory: str):
        os.makedirs(os.path.dirname(to_directory), exist_ok=True)
        os.path.join(to_directory, "")
        with open(to_directory, "w", encoding="utf-8") as f:
            json.dump({
                "epsilon": self._epsilon
            }, f)

    def load(self, from_path: str):
        with open(from_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._epsilon = data["epsilon"]

class DecreasingEpsilonGreedy(EpsilonGreedy):
    """Linearly decreasing epsilon greedy"""

    def __init__(
        self,
        epsilon: float = 1.0,
        decrease_amount: float = 1e-4,
        min_eps: float = 1e-2
    ) -> None:
        super().__init__(epsilon)
        self._decrease_amount = decrease_amount
        self._min_epsilon = min_eps

    def update(self):
        self._epsilon = max(self._epsilon - self._decrease_amount, self._min_epsilon)

    def save(self, to_path: str):
        os.makedirs(os.path.dirname(to_path), exist_ok=True)
        with open(to_path, "w", encoding="utf-8") as f:
            json.dump({
                "epsilon": self._epsilon,
                "decrease_amount": self._decrease_amount,
                "min_epsilon": self._min_epsilon
            }, f)
    
    def load(self, from_path: str):
        with open(from_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._epsilon = data["epsilon"]
            self._decrease_amount = data["decrease_amount"]
            self._min_epsilo = data["min_epsilon"]


class ArgMax(Policy):
    """Exploiting the strategy"""
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        qvalues[available_actions == 0.] = -float("inf")
        actions = qvalues.argmax(-1)
        return actions
    
    def save(self, to_path: str):
        return

    def load(self, from_path: str):
        return
