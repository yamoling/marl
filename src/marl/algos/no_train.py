from dataclasses import dataclass
from typing import cast

from marlenv import ContinuousMARLEnv, DiscreteMARLEnv, MARLEnv

from marl.models.agent import Agent
from marl.models.trainer import Trainer


@dataclass
class NoTrain(Trainer):
    def __init__(self, env: MARLEnv | None = None):
        super().__init__()
        self.env = env

    @staticmethod
    def discrete(env: DiscreteMARLEnv) -> "NoTrain":
        return NoTrain(env)

    @staticmethod
    def continuous(env: ContinuousMARLEnv) -> "NoTrain":
        return NoTrain(env)

    def make_agent(self):
        from marl.agents import RandomAgent

        if self.env is None:
            raise NotImplementedError("Cannot create a random agent without an environment.")
        return cast(Agent, RandomAgent(self.env))
