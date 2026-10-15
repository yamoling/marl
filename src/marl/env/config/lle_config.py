import os
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import lle
from lle.env.builder import Builder
from lle.env.reward_strategy import SingleObjective
from lle.observations import ObservationTypeLiteral

from ..lle_crossing_shaping import LaserCrossingShaping, LaserCrossingSubgoals
from .env_config import EnvConfig


@dataclass
class LLEConfig(EnvConfig[lle.LLE]):
    level_or_path: Literal[1, 2, 3, 4, 5, 6] | str
    """A level or a file path"""
    _: KW_ONLY
    obs_type: ObservationTypeLiteral = "layered"
    state_type: ObservationTypeLiteral = "flattened"
    pbrs: Literal["individual", "joint"] | None = None
    pbrs_subgoals: Literal["laser", "crossing"] = "laser"
    """
    - `laser`: LLE's built-in PBRS, where standing on a beam completes its subgoal (also for the agent that blocks
      it, and for an agent that dies in it).
    - `crossing`: `LaserCrossingShaping`, where living agents are rewarded for entering each beam and for reaching
      its far side.
    """
    time_limit: int | None = -1
    """If <= 0, set to width * height // 2."""

    def __post_init__(self):
        if self.time_limit is not None and self.time_limit <= -1:
            env = self.make_base_env()
            self.time_limit = env.width * env.height // 2
        super().__post_init__()

    def make_base_env(self):
        """
        Build the LLE environment with the requested reward shaping. Falsy `pbrs` values (including the `False` of
        legacy configurations) disable the shaping.

        @ai-edited
        """
        match self.level_or_path:
            case int(level):
                world, name = lle.World.level(level), f"LLE-lvl{level}"
            case str(path):
                world, name = lle.World.from_file(path), f"LLE-{os.path.basename(path)}"
            case other:
                raise NotImplementedError(f"Invalid LLE map: {other}")
        builder = Builder(world).name(name).obs_type(self.obs_type).state_type(self.state_type)
        if not self.pbrs:
            return builder.build()
        lasers_to_reward = None
        if self.level_or_path == 6:
            lasers_to_reward = [(4, 0), (6, 12)]
        match self.pbrs_subgoals:
            case "laser":
                return builder.pbrs(reward_value=1.0, gamma=1.0, lasers_to_reward=lasers_to_reward, with_extras=self.pbrs).build()
            case "crossing":
                sources = world.laser_sources if lasers_to_reward is None else [world.source_at(pos) for pos in lasers_to_reward]
                # Two subgoals per laser, hence half the reward of the "laser" subgoals for the same total shaping.
                shaping = LaserCrossingShaping(SingleObjective(world.n_agents), world, sources, reward_value=1)
                builder = builder.add_extras(LaserCrossingSubgoals(shaping, joint=self.pbrs == "joint"))
                env = builder.name(f"{name}-CrossingPBRS").build()
                env.reward_strategy = shaping
                return env
        raise ValueError(f"Unknown pbrs_subgoals: {self.pbrs_subgoals}")
