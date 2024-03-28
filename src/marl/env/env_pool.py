from rlenv import RLEnv
from rlenv.wrappers import RLEnvWrapper


class EnvPool(RLEnvWrapper):
    def __init__(self, envs: list[RLEnv]):
        obs_shape = envs[0].observation_shape
        act_shape = envs[0].action_space
        state_shape = envs[0].state_shape
        extras_shape = envs[0].extra_feature_shape
        assert all(env.observation_shape == obs_shape for env in envs)
        assert all(env.action_space == act_shape for env in envs)
        assert all(env.state_shape == state_shape for env in envs)
        assert all(env.extra_feature_shape == extras_shape for env in envs)
        super().__init__(envs[0])
        self.envs = envs
        self.t = 0

    def reset(self):
        self.wrapped = self.envs[self.t % len(self.envs)]
        self.t += 1
        return self.wrapped.reset()

    def seed(self, seed_value: int):
        self.t = seed_value
        self.wrapped = self.envs[self.t % len(self.envs)]
        return super().seed(seed_value)
