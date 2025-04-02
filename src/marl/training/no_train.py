from marl.agents import Agent, RandomAgent
from marl.models.trainer import Trainer
from marlenv import MARLEnv, ActionSpace


class NoTrain[A, AS: ActionSpace](Trainer):
    def __init__(self, env: MARLEnv[A, AS]):
        super().__init__()
        self.env = env

    def make_agent(self) -> Agent:
        return RandomAgent(self.env)
