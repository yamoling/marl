from marlenv import MARLEnv, Observation, Space
from marl.models import Agent, Action
from dataclasses import dataclass
import numpy as np


@dataclass
class RandomAgent[S: Space[np.ndarray], T](Agent[T]):
    def __init__(self, env: MARLEnv[S]):
        super().__init__()
        self.env = env

    def choose_action(self, observation: Observation, *, with_details: bool = False) -> Action[T]:
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
