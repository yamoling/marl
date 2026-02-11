from marlenv import MARLEnv, Observation
from marl.models.agent import Agent
from dataclasses import dataclass


@dataclass
class RandomAgent(Agent):
    def __init__(self, env: MARLEnv):
        super().__init__()
        self.env = env

    def choose_action(self, observation: Observation):
        return self.env.action_space.sample(observation.available_actions)

    def value(self, _):
        return 0.0

    def to(self, *args, **kwargs):
        return self

    def set_training(self):
        return

    def set_testing(self):
        return

    def save(self, *args, **kwargs):
        return

    def load(self, *args, **kwargs):
        return

    def randomize(self, *args, **kwargs):
        return
