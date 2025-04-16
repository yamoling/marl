from marlenv import MARLEnv, RLEnvWrapper
from dataclasses import dataclass


@dataclass
class NoReward(RLEnvWrapper):
    def __init__(self, env: MARLEnv):
        super().__init__(env)
        self.name = f"{env.name}-NoReward"

    def step(self, actions):
        step = super().step(actions)
        step.reward.fill(0.0)
        return step
