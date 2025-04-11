from marlenv import MARLEnv, RLEnvWrapper
from typing import Any
from dataclasses import dataclass


@dataclass
class NoReward(RLEnvWrapper[Any, Any]):
    def __init__(self, env: MARLEnv[Any, Any]):
        super().__init__(env)

    def step(self, actions):
        step = super().step(actions)
        step.reward.fill(0.0)
        return step
