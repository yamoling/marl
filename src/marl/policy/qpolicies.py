import random
import numpy as np

from .policy import Policy

class SoftmaxPolicy(Policy):
    """Softmax policy"""

    def __init__(self, n_actions: int, tau: float = 1.):
        self._actions = np.arange(n_actions, dtype=np.int64)
        self._tau = tau
        """Temperature parameter"""

    def get_action(self, qvalues: np.ndarray[np.float32], available_actions: np.ndarray[np.float32]) -> np.ndarray[np.int64]:
        qvalues[available_actions == 0.] = -np.inf
        exp = np.exp(qvalues / self._tau)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        print(exp, probs)
        chosen_actions = [np.random.choice(self._actions, p=agent_probs) for agent_probs in probs]
        return np.array(chosen_actions)

    def summary(self) -> dict[str,]:
        return {
            **super().summary(),
            "tau": self._tau,
            "n_actions": len(self._actions)
        }
    
    @classmethod
    def from_summary(cls, summary: dict[str,]):
        return SoftmaxPolicy(summary["n_actions"], summary["tau"])


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
    
    def summary(self) -> dict[str,]:
        return { **super().summary(), "epsilon": self._epsilon }

    @classmethod
    def from_summary(cls, summary: dict[str,]):
        return EpsilonGreedy(summary["epsilon"])


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

    def summary(self):
        return {
            **super().summary(),
            "epsilon": self._epsilon,
            "decrease_amount": self._decrease_amount,
            "min_epsilon": self._min_epsilon
        }
    
    @classmethod
    def from_summary(cls, summary: dict[str,]):
        return DecreasingEpsilonGreedy(summary["epsilon"], summary["decrease_amount"], summary["min_epsilon"])


class ArgMax(Policy):
    """Exploiting the strategy"""
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        qvalues[available_actions == 0.] = -float("inf")
        actions = qvalues.argmax(-1)
        return actions
    
    @classmethod
    def from_summary(cls, summary: dict[str,]):
        return ArgMax()
