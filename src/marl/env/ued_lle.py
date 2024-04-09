from rlenv import DiscreteSpace, RLEnv
from lle import World


class UED_LLE(RLEnv):
    def __init__(self, width: int, height: int, num_agents: int):
        action_space = DiscreteSpace()
        super().__init__(action_space, observation_shape, state_shape, extra_feature_shape, reward_space)
