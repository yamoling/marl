from rlenv.wrappers import RLEnvWrapper


class ZeroPunishment(RLEnvWrapper):
    def step(self, actions):
        obs, reward, done, truncated, info = super().step(actions)
        reward[reward < 0] = 0
        return obs, reward, done, truncated, info
