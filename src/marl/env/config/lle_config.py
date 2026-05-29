from dataclasses import KW_ONLY, dataclass
from typing import Literal

import lle
from lle.observations import ObservationTypeLiteral

from .env_config import EnvConfig


@dataclass
class LLEConfig(EnvConfig[lle.LLE]):
    level_or_path: Literal[1, 2, 3, 4, 5, 6] | str
    """A level or a file path"""
    _: KW_ONLY
    obs_type: ObservationTypeLiteral = "layered"
    state_type: ObservationTypeLiteral = "flattened"
    pbrs: bool = False
    time_limit: int | None = -1
    """If <= 0, set to width * height // 2."""

    def __post_init__(self):
        if self.time_limit is not None and self.time_limit <= -1:
            env = self.make_base_env()
            self.time_limit = env.width * env.height // 2
        super().__post_init__()

    def make_base_env(self):
        match self.level_or_path:
            case int(level):
                lle_builder = lle.level(level)
            case str(path):
                lle_builder = lle.from_file(path)
            case other:
                raise NotImplementedError(f"Invalid LLE map: {other}")
        builder = lle_builder.obs_type(self.obs_type).state_type(self.state_type)
        if self.pbrs:
            lasers_to_reward = None
            if self.level_or_path == 6:
                lasers_to_reward = [(4, 0), (6, 12)]
            builder = builder.pbrs(reward_value=1.0, gamma=1.0, lasers_to_reward=lasers_to_reward)
        return builder.build()
