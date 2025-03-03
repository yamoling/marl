from typing import Optional, Sequence
from lle import LLE, Position
import numpy.typing as npt
from marlenv import DiscreteActionSpace
from marlenv.wrappers import RLEnvWrapper
import random


class RandomizedLasers(RLEnvWrapper[Sequence[int] | npt.NDArray, DiscreteActionSpace]):
    def __init__(self, env: LLE, sources: Optional[list[Position]] = None):
        super().__init__(env)
        self.name = f"RandomizedLasers-{self.name}"
        self.world = env.world
        if sources is None:
            self.sources = [source.pos for source in self.world.laser_sources]
        else:
            self.sources = sources

    def reset(self):
        for pos in self.sources:
            self.world.source_at(pos).set_colour(random.randint(0, self.n_agents - 1))
        return super().reset()

    def seed(self, seed_value: int):
        random.seed(seed_value)
        return super().seed(seed_value)
