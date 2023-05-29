import random
import numpy as np
from .schedule import LinearSchedule


class Policy:
    def choose_action(self, qvalues: np.ndarray, available_actions: np.ndarray) -> list[int]:
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

class EpsilonGreedy(Policy):
    def __init__(self, start_eps: float, min_eps: float, n_steps: int):
        super().__init__()
        self._epsilon = LinearSchedule(start_eps, min_eps, n_steps)

    def update(self):
        self._epsilon.update()
    
    def choose_action(self, qvalues: np.ndarray, available_actions: np.ndarray):
        qvalues[available_actions == 0.] = -np.inf
        chosen_actions = qvalues.argmax(axis=-1)
        replacements = np.array([random.choice(np.nonzero(available)[0]) for available in available_actions])
        r = np.random.random(len(qvalues))
        mask = r < self._epsilon
        chosen_actions[mask] = replacements[mask]
        return chosen_actions

    def get_action(self, qvalues: np.ndarray, available_actions: np.ndarray):
        return self.choose_action(qvalues, available_actions)