import pickle
from abc import abstractmethod
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Literal, cast

import lle
import marlenv
from marlenv import MARLEnv
from marlenv.adapters import SMAC

from marl.utils import Serializable

type LLELevel = Literal[1, 2, 3, 4, 5, 6]
type LLEObsType = Literal["layered", "flattened", "partial3x3", "partial5x5", "partial7x7", "state", "image", "perspective"]


@dataclass
class EnvConfig[E: MARLEnv](Serializable):
    _: KW_ONLY
    name: str = ""
    agent_id: bool = True
    time_limit: int | None = None
    last_action: bool = False
    maven_noise_size: int | None = None
    env: MARLEnv = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.env = self.make()
        if len(self.name) == 0:
            self.name = self.env.name

    @staticmethod
    def from_any[A](
        env: MARLEnv[A],
        agent_id: bool = True,
        time_limit: int | None = None,
        last_action: bool = False,
        maven_noise_size: int | None = None,
        **kwargs,
    ) -> "EnvConfig[MARLEnv[A]]":
        """Create an EnvConfig from any MARLEnv by pickling it."""
        return PickleEnvConfig.create(
            env,
            agent_id=agent_id,
            time_limit=time_limit,
            last_action=last_action,
            maven_noise_size=maven_noise_size,
            **kwargs,
        )

    @abstractmethod
    def make_base_env(self) -> E: ...

    def make(self):
        base_env = self.make_base_env()
        builder = marlenv.Builder(base_env)
        if self.time_limit is not None:
            builder = builder.time_limit(self.time_limit)
        if self.agent_id:
            builder = builder.agent_id()
        if self.last_action:
            builder = builder.last_action()
        if self.maven_noise_size is not None:
            builder = builder.pad("extra", self.maven_noise_size, label="maven")
        return builder.build()

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def n_actions(self):
        return self.env.n_actions

    @property
    def observation_shape(self):
        return self.env.observation_shape

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def state_size(self):
        return self.env.state_size

    @property
    def state_extras_size(self):
        return self.env.state_extras_size

    @property
    def extras_shape(self):
        return self.env.extras_shape

    @property
    def extras_size(self):
        return self.env.extras_size

    @property
    def extras_meanings(self):
        return self.env.extras_meanings

    @property
    def n_objectives(self):
        return self.env.n_objectives

    @property
    def reward_space(self):
        return self.env.reward_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def noise_size(self):
        """Convenience property to get the size of the MAVEN noise space without type-checking."""
        if self.maven_noise_size is None:
            raise ValueError("This environment does not have a maven noise space.")
        return self.maven_noise_size

    @property
    def maven_bandit_obs_shape(self):
        """MAVEN's bandit stacks the observations of all agents"""
        return (self.env.observation_shape[0] * self.env.n_agents, *self.env.observation_shape[1:])

    @property
    def maven_bandit_extras_shape(self):
        """MAVEN's bandit stacks the extras of all agents, but removes the noise extras"""
        return ((self.env.extras_size - self.noise_size) * self.env.n_agents,)


@dataclass
class LLEConfig(EnvConfig[lle.LLE]):
    level_or_path: Literal[1, 2, 3, 4, 5, 6] | str
    """A level or a file path"""
    obs_type: LLEObsType = "layered"
    state_type: LLEObsType = "state"
    _: KW_ONLY
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
        return lle_builder.obs_type(self.obs_type).state_type(self.state_type).build()


@dataclass
class SMACConfig(EnvConfig[SMAC]):
    map_name: str
    _: KW_ONLY
    debug: bool = False
    name: str = "SMAC"

    def make_base_env(self):
        return SMAC(self.map_name, debug=self.debug)


@dataclass
class PickleEnvConfig(EnvConfig[MARLEnv]):
    pickle_path: str
    _: KW_ONLY

    def make_base_env(self):
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def create[A](
        cls,
        env: MARLEnv[A],
        agent_id: bool = True,
        time_limit: int | None = None,
        last_action: bool = False,
        maven_noise_size: int | None = None,
        **kwargs,
    ) -> EnvConfig[MARLEnv[A]]:
        env_dir = Path("envs")
        env_dir.mkdir(exist_ok=True)
        env_file = env_dir / f"{env.name}.pkl"
        suffix = 0
        while env_file.exists():
            with open(env_file, "rb") as f:
                other = cast(MARLEnv[A], pickle.load(f))
            if other == env:
                # If we find a matching environment, just reuse the same pickle file.
                return cls(
                    pickle_path=f.name,
                    agent_id=agent_id,
                    time_limit=time_limit,
                    last_action=last_action,
                    maven_noise_size=maven_noise_size,
                    **kwargs,
                )
            suffix += 1
            env_file = env_dir / f"{env.name}-{suffix}.pkl"
        with open(env_file, "wb") as f:
            pickle.dump(env, f)
        return cls(
            pickle_path=f.name,
            agent_id=agent_id,
            time_limit=time_limit,
            last_action=last_action,
            maven_noise_size=maven_noise_size,
            **kwargs,
        )
