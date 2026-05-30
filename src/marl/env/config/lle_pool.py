import os
from dataclasses import KW_ONLY, dataclass

import marlenv
import numpy as np
import numpy.typing as npt
from lle import World
from lle.observations import ObservationTypeLiteral
from marlenv.catalog import EnvPool

from .env_config import EnvConfig


@dataclass
class LLEPool(EnvConfig[EnvPool[npt.NDArray[np.int64]]]):
    directory: str
    size: int
    _: KW_ONLY
    offset: int = 0
    width: int = 13
    height: int = 12
    n_lasers: int = 3
    _n_agents: int = 4
    obs_type: ObservationTypeLiteral = "layered"
    state_type: ObservationTypeLiteral = "flattened"
    time_limit: int | None = -1
    """If <= 0, set to width * height // 2."""

    def __post_init__(self):
        if self.time_limit is not None and self.time_limit <= -1:
            self.time_limit = self.width * self.height // 2
        super().__post_init__()

    def make_base_env(self):
        from lle import ObservationType
        from lle.env import LLE, SingleObjective

        files = sorted(os.listdir(self.directory))[self.offset : self.offset + self.size]
        assert len(files) == self.size
        files = [os.path.join(self.directory, f) for f in files]
        w = World.from_file(files[0])
        envs = [
            LLE(
                World.from_file(f),
                SingleObjective(w.n_agents),
                ObservationType.from_str(self.obs_type),
                ObservationType.from_str(self.state_type),
                name=f.replace("/", "-"),
            )
            for f in files
        ]
        return EnvPool(envs)

    def to_dict(self):
        return super().to_dict()

    def __hash__(self):
        return id(self)
