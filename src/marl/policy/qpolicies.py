import random
from typing import Any
import numpy as np
from dataclasses import dataclass

from marl.utils import schedule

from .policy import Policy

@dataclass
class SoftmaxPolicy(Policy):
    """Softmax policy"""
    n_actions: int
    tau: float = 1.

    def __post_init__(self):
        self.actions = np.arange(self.n_actions, dtype=np.int64)

    def get_action(self, qvalues: np.ndarray[np.float32], available_actions: np.ndarray[np.float32]) -> np.ndarray[np.int64]:
        qvalues[available_actions == 0.] = -np.inf
        exp = np.exp(qvalues / self.tau)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        chosen_actions = [np.random.choice(self.actions, p=agent_probs) for agent_probs in probs]
        return np.array(chosen_actions)


@dataclass
class EpsilonGreedy(Policy):
    """Epsilon Greedy policy"""
    epsilon: schedule.Schedule

    @classmethod
    def linear(cls, start_eps: float, min_eps: float, n_steps: int):
        return cls(schedule.LinearSchedule(start_eps, min_eps, n_steps))
    
    @classmethod
    def exponential(cls, start_eps: float, min_eps: float, n_steps: float):
        return cls(schedule.ExpSchedule(start_eps, min_eps, n_steps))
    
    @classmethod
    def constant(cls, eps: float):
        return cls(schedule.ConstantSchedule(eps))

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> np.ndarray:
        qvalues[available_actions == 0.] = -np.inf
        chosen_actions = qvalues.argmax(axis=-1)
        replacements = np.array([random.choice(np.nonzero(available)[0]) for available in available_actions])
        r = np.random.random(len(qvalues))
        mask = r < self.epsilon
        chosen_actions[mask] = replacements[mask]
        return chosen_actions
    
    def update(self):
        return self.epsilon.update()
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        data["epsilon"] = schedule.from_dict(data["epsilon"])
        return super().from_dict(data)


@dataclass
class ArgMax(Policy):
    """Exploiting the strategy"""
    def __init__(self):
        super().__init__()

    def get_action(self, qvalues: np.ndarray, _) -> np.ndarray:
        actions = qvalues.argmax(-1)
        return actions
    