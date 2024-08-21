import rlenv
from .algo import RLAlgo
from typing import TypeVar

A = TypeVar("A", bound=rlenv.ActionSpace)


class RandomAlgo(RLAlgo):
    def __init__(self, env: rlenv.RLEnv[A]):
        super().__init__()
        self.env = env

    def choose_action(self, obs: rlenv.Observation):
        return self.env.action_space.sample(obs.available_actions)

    def value(self, _):
        return 0.0

    def to(self, _):
        return self

    def set_training(self):
        return

    def set_testing(self):
        return

    def save(self, _):
        return

    def load(self, _):
        return

    def randomize(self):
        return
