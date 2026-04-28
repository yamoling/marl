from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch


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

    def to_one_hot(self):
        return OneHotBandit(self)


class OneHotBandit(ContextualBandit[npt.NDArray[np.int64]]):
    def __init__(self, bandit: CategoricalBandit):
        self.bandit = bandit

    def choose_action(self, /, **kwargs) -> npt.NDArray[np.int64]:
        action = self.bandit.choose_action(**kwargs)
        one_hot = np.zeros(self.bandit.n_actions, dtype=np.int64)
        one_hot[action] = 1
        return one_hot
