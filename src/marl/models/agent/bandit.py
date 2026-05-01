from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
from marlenv import DiscreteSpace


class ContextualBandit[T](ABC):
    """Contextual Bandit. Differs from bandits because it accepts an input."""

    @abstractmethod
    def choose_action(self, /, **kwargs) -> T:
        """
        Choose an action for each agent.

        There can be some input data
        """

    def to(self, device: torch.device):
        return self


class CategoricalBandit(ContextualBandit[int]):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.space = DiscreteSpace(n_actions)

    def to_one_hot(self):
        return OneHotBandit(self)


class OneHotBandit(ContextualBandit[npt.NDArray[np.float32]]):
    def __init__(self, bandit: CategoricalBandit):
        self.bandit = bandit

    @property
    def n_actions(self):
        return self.bandit.n_actions

    def choose_action(self, /, **kwargs) -> npt.NDArray[np.float32]:
        action = self.bandit.choose_action(**kwargs)
        one_hot = np.zeros(self.bandit.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot
