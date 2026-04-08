from marlenv import MARLEnv, Observation, Space
from marl.models import Agent, Action
from dataclasses import dataclass
from typing import TypeVar, Generic
import numpy as np

S = TypeVar("S", bound=Space[np.ndarray])


@dataclass
class RandomAgent(Generic[S], Agent):
    def __init__(self, env: MARLEnv[S]):
        super().__init__()
        self.env = env

    def choose_action(self, observation: Observation):
        return Action(self.env.action_space.sample(observation.available_actions))

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
