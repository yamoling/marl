from marl.agents import Agent, RandomAgent
from marl.models.trainer import Trainer
from marlenv import MARLEnv, Space
from dataclasses import dataclass


@dataclass
class NoTrain[A: Space](Trainer):
    def __init__(self, env: MARLEnv[A]):
        super().__init__()
        self.env = env

    def make_agent(self) -> Agent:
        return RandomAgent(self.env)
