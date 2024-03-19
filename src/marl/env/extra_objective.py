from typing import Iterable, Optional
import numpy as np
from rlenv.models import DiscreteSpace
from rlenv.wrappers import RLEnv, RLEnvWrapper


class ExtraObjective(RLEnvWrapper):
    def __init__(self, env: RLEnv, n_objectives: int, fake_rewards: Optional[Iterable[float]] = None):
        self.n_objectives = n_objectives
        labels = env.reward_space.labels + [f"Fake objective {i}" for i in range(n_objectives)]
        reward_space = DiscreteSpace(env.reward_space.size + n_objectives, labels)
        super().__init__(env, reward_space=reward_space)
        if fake_rewards is None:
            fake_rewards = [0.0] * n_objectives
        else:
            fake_rewards = list(fake_rewards)
            assert len(fake_rewards) == n_objectives
        self.fake_rewards = np.array(fake_rewards)

    def step(self, actions: list[int] | np.ndarray):
        obs, reward, done, truncated, info = super().step(actions)
        reward = np.concatenate([reward, self.fake_rewards])
        return obs, reward, done, truncated, info
