from marlenv import MARLEnv, RLEnvWrapper, Space
from dataclasses import dataclass


@dataclass
class NoReward[S: Space](RLEnvWrapper[S]):
    def __init__(self, env: MARLEnv[S]):
        super().__init__(env)
        self.name = f"{env.name}-NoReward"

    def step(self, actions):
        step = super().step(actions)
        step.reward.fill(0.0)
        return step
