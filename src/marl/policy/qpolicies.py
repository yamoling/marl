import random
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from marl.utils import schedule
from marl.models import Policy


@dataclass
class SoftmaxPolicy(Policy):
    """Softmax policy"""

    tau: float

    def __init__(self, n_actions: int, tau: float = 1.0):
        super().__init__()
        self.actions = np.arange(n_actions, dtype=np.int64)
        self.tau = tau

    def get_action(self, qvalues: npt.NDArray[np.float32], available_actions: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        qvalues[available_actions == 0.0] = -np.inf
        exp = np.exp(qvalues / self.tau)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        chosen_actions = [np.random.choice(self.actions, p=agent_probs) for agent_probs in probs]
        return np.array(chosen_actions)

    def update(self, _: int) -> dict[str, float]:
        return {"softmax-tau": self.tau}


@dataclass
class EpsilonGreedy(Policy):
    """Epsilon Greedy policy"""

    epsilon: schedule.Schedule

    def __init__(self, epsilon: schedule.Schedule):
        super().__init__()
        self.epsilon = epsilon

    @classmethod
    def linear(cls, start_eps: float, end_eps: float, n_steps: int):
        return cls(schedule.LinearSchedule(start_eps, end_eps, n_steps))

    @classmethod
    def exponential(cls, start_eps: float, end_eps: float, n_steps: int):
        return cls(schedule.ExpSchedule(start_eps, end_eps, n_steps))

    @classmethod
    def constant(cls, eps: float):
        return cls(schedule.ConstantSchedule(eps))

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray, qvalues_cp: np.ndarray = None) -> np.ndarray:
        qvalues[available_actions == 0.0] = -np.inf
        chosen_actions = qvalues.argmax(axis=-1)
        r = np.random.random(len(qvalues))
        if qvalues_cp is not None: qvalues[...] = np.take_along_axis(qvalues,chosen_actions[:,None],-1) # Need to scale when MO?
        replacements = np.array([random.choice(np.nonzero(available)[0]) for available in available_actions])
        mask = r < self.epsilon
        chosen_actions[mask] = replacements[mask]
        if qvalues_cp is not None: qvalues_cp[...] = np.take_along_axis(qvalues_cp,chosen_actions[:,None],-1) # Need to scale when MO?
        return chosen_actions

    def update(self, step_num: int):
        self.epsilon.update(step_num)
        return {"epsilon": self.epsilon.value}


@dataclass
class ArgMax(Policy):
    """Exploiting the strategy"""

    def __init__(self):
        super().__init__()

    def get_action(self, qvalues: np.ndarray, available_actions: npt.NDArray[np.float32]) -> np.ndarray:
        qvalues[available_actions == 0.0] = -np.inf
        actions = qvalues.argmax(-1)
        qvalues[...] = np.take_along_axis(qvalues,actions[:,None],-1) # Need to scale when MO?
        return actions

    def update(self, step_num: int):
        return {}
