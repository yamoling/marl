import marlenv
from .algo import RLAlgo
from typing import TypeVar

A = TypeVar("A", bound=marlenv.ActionSpace)


class RandomAlgo(RLAlgo):
    def __init__(self, env: marlenv.RLEnv[A]):
        super().__init__()
        self.env = env

    def choose_action(self, obs: marlenv.Observation):
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
