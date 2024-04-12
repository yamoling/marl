from rlenv.wrappers import RLEnvWrapper
from lle import LLE
from itertools import product
import random


class RandomInitialPos(RLEnvWrapper):
    def __init__(self, env: LLE, min_i: int, max_i: int, min_j: int, max_j: int):
        super().__init__(env)
        self.min_i = min_i
        self.min_j = min_j
        self.world = env.world
        self.lle = env
        self.ALL_INITIAL_POS = list(product(range(min_i, max_i + 1), range(min_j, max_j + 1)))

    def reset(self):
        super().reset()
        state = self.world.get_state()
        state.agents_positions = random.sample(self.ALL_INITIAL_POS, k=self.n_agents)
        self.world.set_state(state)
        return self.lle.get_observation()

    def seed(self, seed_value: int):
        random.seed(seed_value)
        return super().seed(seed_value)
