from dataclasses import KW_ONLY, asdict, dataclass, field
from typing import Any, Literal, cast

import marlenv
from marlenv import MARLEnv

from .config import Config

type LLELevel = Literal[1, 2, 3, 4, 5, 6]
type LLEObsType = Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]


@dataclass
class EnvConfig(Config[MARLEnv]):
    _: KW_ONLY
    agent_id: bool = True
    time_limit: int | None = None
    last_action: bool = False


@dataclass
class LLEEnvConf:
    level: LLELevel
    kind: Literal["lle"] = field(default="lle", init=False)
    obs_type: LLEObsType = "layered"
    state_type: LLEObsType = "layered"
    agent_id: bool = True
    last_action: bool = False
    time_limit: int | None = None

    def make(self) -> MARLEnv:
        from lle import LLE

        lle = LLE.level(self.level).obs_type(self.obs_type).state_type(self.state_type).build()
        builder = marlenv.Builder(lle)
        if self.time_limit is not None:
            builder = builder.time_limit(self.time_limit)
        if self.agent_id:
            builder = builder.agent_id()
        if self.last_action:
            builder = builder.last_action()
        return builder.build()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLEEnvConf":
        level = int(data["level"])
        if level not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"Invalid LLE level: {level}. Expected one of 1..6")

        obs_type = str(data.get("obs_type", "layered"))
        state_type = str(data.get("state_type", "layered"))
        allowed_types = {"layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"}
        if obs_type not in allowed_types:
            raise ValueError(f"Invalid LLE observation type: {obs_type}")
        if state_type not in allowed_types:
            raise ValueError(f"Invalid LLE state type: {state_type}")

        return cls(
            level=cast(LLELevel, level),
            obs_type=cast(LLEObsType, obs_type),
            state_type=cast(LLEObsType, state_type),
            agent_id=bool(data.get("agent_id", True)),
            last_action=bool(data.get("last_action", False)),
            time_limit=data.get("time_limit"),
        )


@dataclass
class SMACEnvConf:
    map_name: str
    kind: Literal["smac"] = field(default="smac", init=False)
    debug: bool = False
    agent_id: bool = True
    last_action: bool = True

    def make(self) -> MARLEnv:
        env = marlenv.adapters.SMAC(self.map_name, debug=self.debug)
        builder = marlenv.Builder(env)
        if self.agent_id:
            builder = builder.agent_id()
        if self.last_action:
            builder = builder.last_action()
        return builder.build()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SMACEnvConf":
        return cls(
            map_name=str(data["map_name"]),
            debug=bool(data.get("debug", False)),
            agent_id=bool(data.get("agent_id", True)),
            last_action=bool(data.get("last_action", True)),
        )


type EnvConf = LLEEnvConf | SMACEnvConf


def env_conf_from_dict(data: dict[str, Any]) -> EnvConf:
    kind = data.get("_")
    match kind:
        case "lle":
            return LLEEnvConf.from_dict(data)
        case "smac":
            return SMACEnvConf.from_dict(data)
        case other:
            raise ValueError(f"Unknown environment configuration kind: {other}")
