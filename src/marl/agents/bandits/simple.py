import numpy as np
import numpy.typing as npt
from marlenv import DiscreteSpace, Observation

from marl.models import Action, Agent


class CategoricalBandit(Agent[npt.NDArray[np.int64]]):
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.space = DiscreteSpace(n_actions)

    def to_one_hot(self):
        return OneHotBandit(self)


class OneHotBandit(Agent[npt.NDArray[np.float32]]):
    def __init__(self, bandit: CategoricalBandit):
        self.bandit = bandit

    @property
    def n_actions(self):
        return self.bandit.n_actions

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        wrapped_action = self.bandit.choose_action(observation, with_details=with_details)
        action = wrapped_action.action.item()
        one_hot = np.zeros(self.bandit.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        return Action(one_hot, **wrapped_action.details)
