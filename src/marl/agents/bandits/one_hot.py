from dataclasses import dataclass

import numpy as np
from marlenv import DiscreteSpace

from marl.models import Bandit


@dataclass
class UniformOneHot(Bandit):
    n_actions: int

    def __post_init__(self):
        self._action_space = DiscreteSpace(self.n_actions)

    def choose_action(self, /, **kwargs):
        zeros = np.zeros(self.n_actions, dtype=np.int64)
        zeros[self._action_space.sample()] = 1
        return zeros


class DeepOneHot(Bandit):
    def __init__(self):
        super().__init__()
