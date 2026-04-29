from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from marlenv import ContinuousSpace, DiscreteSpace, MARLEnv, MultiDiscreteSpace

from marl.agents import RandomAgent
from marl.models.agent import Agent
from marl.models.trainer import Trainer


@dataclass
class NoTrain[T](Trainer[T]):
    def __init__(self, env: MARLEnv):
        super().__init__()
        self.env = env

    @staticmethod
    def discrete(env: MARLEnv[MultiDiscreteSpace] | MARLEnv[DiscreteSpace]) -> "NoTrain[npt.NDArray[np.int64]]":
        return NoTrain(env)

    @staticmethod
    def continuous(env: MARLEnv[ContinuousSpace]) -> "NoTrain[npt.NDArray[np.float32]]":
        return NoTrain(env)

    def make_agent(self) -> Agent:
        return RandomAgent(self.env)
