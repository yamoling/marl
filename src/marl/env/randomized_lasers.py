import random
from lle import LLE
from rlenv.wrappers import RLEnvWrapper


class RandomizedLasers(RLEnvWrapper):
    def __init__(self, lle: LLE):
        super().__init__(lle)
        self.world = lle.world
        self.laser_sources = list(self.world.laser_sources.values())

    def reset(self):
        for source in self.laser_sources:
            self.world.set_laser_colour(source, random.randint(0, self.n_agents - 1))
        return super().reset()

    def seed(self, seed_value: int):
        random.seed(seed_value)
        return super().seed(seed_value)
