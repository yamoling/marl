import os
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from lle import World
from lle.observations import ObservationTypeLiteral
from marlenv.catalog import EnvPool

from .env_config import EnvConfig


@dataclass
class LLEPool(EnvConfig[EnvPool[npt.NDArray[np.int64]]]):
    directory: str | Literal["mix"] | Path
    size: int
    """Size of the pool"""
    _: KW_ONLY
    sequential: bool = True
    offset: int = 0
    dims: tuple[int, int] | None = None
    obs_type: ObservationTypeLiteral = "layered"
    state_type: ObservationTypeLiteral = "flattened"
    time_limit: int | None = -1
    """If <= 0, set to width * height // 2."""

    def __post_init__(self):
        if self.time_limit is not None and self.time_limit <= -1:
            if self.dims is None:
                raise ValueError("dims must be specified if time_limit is not set.")
            self.time_limit = self.dims[0] * self.dims[1] // 2
        super().__post_init__()

    def make_base_env(self):
        from lle import ObservationType
        from lle.env import LLE, SingleObjective

        files = sorted(os.listdir(self.directory))[self.offset : self.offset + self.size]
        files = [os.path.join(self.directory, f) for f in files]
        assert len(files) == self.size
        assert len(set(files)) == len(files)
        w = World.from_file(files[0])
        for f in files:
            print(f)
            World.from_file(f)
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
        return EnvPool(envs, self.sequential)

    def to_dict(self):
        return super().to_dict()

    def __hash__(self):
        return id(self)
