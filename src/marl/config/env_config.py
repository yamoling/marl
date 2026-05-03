from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import lle
import marlenv
from marlenv import MARLEnv, MultiDiscreteSpace, Space

from .config import Config

type LLELevel = Literal[1, 2, 3, 4, 5, 6]
type LLEObsType = Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]


@dataclass
class EnvConfig[AS: Space](Config[MARLEnv[AS]]):
    _: KW_ONLY
    agent_id: bool = True
    time_limit: int | None = None
    last_action: bool = False

    @abstractmethod
    def make_base_env(self) -> MARLEnv[AS]: ...

    def make(self):
        base_env = self.make_base_env()
        builder = marlenv.Builder(base_env)
        if self.time_limit is not None:
            builder = builder.time_limit(self.time_limit)
        if self.agent_id:
            builder = builder.agent_id()
        if self.last_action:
            builder = builder.last_action()
        return builder.build()


@dataclass
class LLEConfig(EnvConfig[MultiDiscreteSpace]):
    level_or_path: Literal[1, 2, 3, 4, 5, 6] | str
    """A level or a file path"""
    obs_type: LLEObsType = "layered"
    state_type: LLEObsType = "state"
    _: KW_ONLY
    time_limit: int | None = -1
    """If <= 0, set to width * height // 2."""

    def __post_init__(self):
        if self.time_limit is None:
            return
        if self.time_limit <= -1:
            env = self.make_base_env()
            self.time_limit = env.width * env.height // 2

    def make_base_env(self):
        match self.level_or_path:
            case int(level):
                lle_builder = lle.level(level)
            case str(path):
                lle_builder = lle.from_file(path)
            case other:
                raise NotImplementedError(f"Invalid LLE map: {other}")
        return lle_builder.obs_type(self.obs_type).state_type(self.state_type).build()


@dataclass
class SMACConfig(EnvConfig[MultiDiscreteSpace]):
    map_name: str
    debug: bool = False

    def make_base_env(self):
        return marlenv.adapters.SMAC(self.map_name, debug=self.debug)
