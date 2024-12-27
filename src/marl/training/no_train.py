from marl.agents import Agent, RandomAgent
from marl.models.trainer import Trainer
from marlenv import MARLEnv, ActionSpace


class NoTrain[A, AS: ActionSpace](Trainer):
    def __init__(self, env: MARLEnv[A, AS]):
        super().__init__("episode")
        self.env = env

    def to(self, _):
        return self

    def randomize(self):
        return

    def make_agent(self) -> Agent:
        return RandomAgent(self.env)
