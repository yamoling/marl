from marlenv import MARLEnv, Observation, ActionSpace
from .agent import Agent


class RandomAgent[A, AS: ActionSpace](Agent[A]):
    def __init__(self, env: MARLEnv[A, AS]):
        super().__init__()
        self.env = env

    def choose_action(self, obs: Observation) -> A:
        return self.env.action_space.sample(obs.available_actions)  # type: ignore

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
