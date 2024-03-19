from rlenv.wrappers import RLEnvWrapper
from rlenv import DiscreteSpace
from lle import LLE


class LLEShapingWrapper(RLEnvWrapper):
    def __init__(self, env: LLE):
        super().__init__(env)
        self.world = env.world
        env.reward_space
