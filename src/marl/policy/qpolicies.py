import random
import numpy as np
from dataclasses import dataclass
from serde import serde

from marl.utils import schedule

from .policy import Policy

@serde
@dataclass
class SoftmaxPolicy(Policy):
    """Softmax policy"""
    tau: float

    def __init__(self, n_actions: int, tau: float = 1.):
        super().__init__()
        self.actions = np.arange(n_actions, dtype=np.int64)
        self.tau = tau

    def get_action(self, qvalues: np.ndarray[np.float32], available_actions: np.ndarray[np.float32]) -> np.ndarray[np.int64]:
        qvalues[available_actions == 0.] = -np.inf
        exp = np.exp(qvalues / self.tau)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        chosen_actions = [np.random.choice(self.actions, p=agent_probs) for agent_probs in probs]
        return np.array(chosen_actions)

@serde
@dataclass
class EpsilonGreedy(Policy):
    """Epsilon Greedy policy"""
    epsilon: schedule.Schedule

    def __init__(self, epsilon: schedule.Schedule):
        super().__init__()
        self.epsilon = epsilon

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
        r = np.random.random(len(qvalues))
        replacements = np.array([random.choice(np.nonzero(available)[0]) for available in available_actions])
        mask = r < self.epsilon
        chosen_actions[mask] = replacements[mask]
        return chosen_actions
    
    def update(self, time_step: int):
        self.epsilon.update(time_step)
    
@serde
@dataclass
class ArgMax(Policy):
    """Exploiting the strategy"""

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray[np.float32]) -> np.ndarray:
        qvalues[available_actions == 0.] = -np.inf
        actions = qvalues.argmax(-1)
        return actions
    