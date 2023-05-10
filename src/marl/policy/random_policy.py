import numpy as np

from .policy import Policy


class RandomPolicy(Policy):
    """Random policy. Take actions randomly"""

    def __init__(self, n_actions: int, n_agents: int) -> None:
        super().__init__()
        self._n_actions = n_actions
        self._n_agents = n_agents

    def choose_action(self, *_observation):
        return np.random.randint(0, self._n_actions, self._n_agents)

    def save(self, _filename: str):
        return

    def load(self, _filename: str):
        return
