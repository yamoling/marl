import random
from dataclasses import KW_ONLY, dataclass, field

import numpy as np
import numpy.typing as npt

from marl.models import Policy
from marl.utils import schedule, tuning


@dataclass
class SoftmaxPolicy(Policy):
    """Softmax policy"""

    n_actions: int
    _: KW_ONLY
    tau: float = field(default=1.0, metadata=tuning(0.01, 100.0, log=True))

    def __post_init__(self):
        super().__post_init__()
        self.actions = np.arange(self.n_actions, dtype=np.int64)

    def get_action(
        self,
        qvalues: npt.NDArray[np.float32],
        available_actions: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.int64]:
        if available_actions is not None:
            qvalues[available_actions == 0.0] = -np.inf
        exp = np.exp(qvalues / self.tau)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)
        chosen_actions = [np.random.choice(self.actions, p=agent_probs) for agent_probs in probs]
        return np.array(chosen_actions)

    def update(self, time_step: int) -> dict[str, float]:
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
        return cls(schedule.LinearSchedule(start_value=start_eps, end_value=end_eps, n_steps=n_steps))

    @classmethod
    def exponential(cls, start_eps: float, end_eps: float, n_steps: int):
        return cls(schedule.ExpSchedule(start_value=start_eps, end_value=end_eps, n_steps=n_steps))

    @classmethod
    def constant(cls, eps: float):
        return cls(schedule.ConstantSchedule(start_value=eps))

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray | None = None) -> np.ndarray:
        if available_actions is not None:
            qvalues[available_actions == 0.0] = -np.inf
        else:
            available_actions = np.full_like(qvalues, True)
        chosen_actions = qvalues.argmax(axis=-1)
        r = np.random.random(len(qvalues))
        replacements = np.array([random.choice(np.nonzero(available)[0]) for available in available_actions])
        mask = r < self.epsilon
        chosen_actions[mask] = replacements[mask]
        return chosen_actions

    def update(self, time_step: int):
        self.epsilon.update(time_step)
        return {"epsilon": self.epsilon.value}

    # @classmethod
    # def from_dict(cls, d: dict):
    #     d = d["epsilon"]
    #     name = d.pop("name")
    #     match name:
    #         case "LinearSchedule":
    #             epsilon = schedule.LinearSchedule(d["start_value"], d["end_value"], d["n_steps"])
    #             return cls(epsilon=epsilon)
    #         case "ExpSchedule":
    #             epsilon = schedule.ExpSchedule(d["start_value"], d["min_value"], d["n_steps"])
    #             return cls(epsilon=epsilon)
    #         case "ConstantSchedule":
    #             epsilon = schedule.ConstantSchedule(d["start_value"])
    #             return cls(epsilon=epsilon)
    #         case other:
    #             raise ValueError(f"Unknown policy type: {other}")


@dataclass
class ArgMax(Policy):
    """Exploiting the strategy"""

    def get_action(self, qvalues: np.ndarray, available_actions: npt.NDArray[np.float32] | None = None) -> np.ndarray:
        if available_actions is not None:
            qvalues[available_actions == 0.0] = -np.inf
        return qvalues.argmax(-1)

    def update(self, time_step: int):
        return {}
