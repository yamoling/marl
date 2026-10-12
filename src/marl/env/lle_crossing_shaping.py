"""
Laser-crossing reward shaping for LLE.

LLE's built-in PBRS (`lle.env.reward_strategy.PotentialShapedLLE`) considers that an agent has completed the
subgoal of a laser as soon as it stands on one of its beam tiles. The agent that blocks a laser therefore
completes its subgoal by blocking, and nothing in the shaped reward incentivizes it to actually cross the laser
afterwards. On level 6, the blocker of the second laser then learns to stay in its beam forever (exit rate of
0.75). Moreover, an agent that dies in a laser receives the subgoal reward, which cancels the death penalty.

`LaserCrossingShaping` rewards, once per episode and per agent, both entering the beam and reaching its far side
(the tiles adjacent to the beam that are on the side of the exits), which mirrors the shaping used in the ECAI 2024
experiments. Only living agents can complete a subgoal.
"""

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from lle import World, WorldEvent, tiles
from lle.env.extras_generators import ExtraGenerator
from lle.env.reward_strategy import RewardStrategy
from lle.types import Position

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _reachable(world: World, sources: Iterable[Position], blocked: set[Position]) -> set[Position]:
    """Positions reachable from `sources` with 4-connectivity, without stepping on `blocked` positions. @ai-generated"""
    visited = {pos for pos in sources if pos not in blocked}
    queue = deque(visited)
    while queue:
        i, j = queue.popleft()
        for di, dj in _NEIGHBOURS:
            pos = (i + di, j + dj)
            if not (0 <= pos[0] < world.height and 0 <= pos[1] < world.width) or pos in blocked or pos in visited:
                continue
            visited.add(pos)
            queue.append(pos)
    return visited


def laser_crossing_subgoals(world: World, source: tiles.LaserSource) -> tuple[set[Position], set[Position]]:
    """
    Return the beam tiles of `source` and the tiles of its far side.

    The far side consists of the walkable tiles adjacent to the beam from which an exit can be reached without
    stepping on the beam again, but from which no start position can. Dead ends behind the laser (which require
    crossing the beam back to reach an exit) are therefore excluded, and a laser that can be bypassed has no far side.

    @ai-generated
    """
    beam = {laser.pos for laser in world.lasers if laser.laser_id == source.laser_id}
    obstacles = set(world.wall_pos) | set(world.void_pos) | {s.pos for s in world.laser_sources}
    exit_side = _reachable(world, world.exit_pos, obstacles | beam)
    start_side = _reachable(world, world.start_pos, obstacles | beam)
    far_side = set[Position]()
    for i, j in beam:
        for di, dj in _NEIGHBOURS:
            pos = (i + di, j + dj)
            if pos in exit_side and pos not in start_side:
                far_side.add(pos)
    return beam, far_side


class LaserCrossingShaping(RewardStrategy):
    """
    Wrap a single-objective reward strategy and add `reward_value` the first time (per episode) each living agent
    enters the beam of each given laser and the first time it reaches the far side of that laser.

    In terms of potential-based shaping, the potential of a state is `reward_value` times the number of subgoals
    completed during the episode, with an undiscounted potential difference and no terminal reset.

    @ai-generated
    """

    def __init__(self, strategy: RewardStrategy, world: World, sources: Sequence[tiles.LaserSource], reward_value: float):
        if strategy.n_objectives != 1:
            raise NotImplementedError("LaserCrossingShaping only supports single-objective reward strategies.")
        super().__init__(strategy.n_agents, strategy.objectives)
        self.strategy = strategy
        self.reward_value = reward_value
        self._world = world
        self.subgoals = list[set[Position]]()
        self.meanings = list[str]()
        for source in sources:
            beam, far_side = laser_crossing_subgoals(world, source)
            if len(far_side) == 0:
                raise ValueError(f"Laser {source.laser_id} at {source.pos} can be bypassed: it has no far side to reward.")
            self.subgoals += [beam, far_side]
            self.meanings += [f"Laser {source.laser_id} at {source.pos} entered", f"Laser {source.laser_id} at {source.pos} crossed"]
        self.reached = np.full((world.n_agents, len(self.subgoals)), False, dtype=np.bool)

    def _update_reached(self) -> int:
        """Mark the subgoals completed by the living agents and return the number of new completions. @ai-generated"""
        n_new = 0
        for agent_num, (agent, pos) in enumerate(zip(self._world.agents, self._world.agents_positions)):
            if agent.is_dead:
                continue
            for subgoal_num, positions in enumerate(self.subgoals):
                if not self.reached[agent_num, subgoal_num] and pos in positions:
                    self.reached[agent_num, subgoal_num] = True
                    n_new += 1
        return n_new

    def compute_reward(self, events: Sequence[WorldEvent]) -> npt.NDArray[np.float32]:
        """@ai-generated"""
        reward = self.strategy.compute_reward(events)
        reward[0] += self._update_reached() * self.reward_value
        self.n_deads = self.strategy.n_deads
        self.n_arrived = self.strategy.n_arrived
        return reward

    def reset(self):
        """Subgoals already satisfied by the initial positions are completed without reward. @ai-generated"""
        super().reset()
        self.strategy.reset()
        self.reached.fill(False)
        self._update_reached()


class LaserCrossingSubgoals(ExtraGenerator):
    """
    Extras that indicate which subgoals of a `LaserCrossingShaping` have been completed. In joint mode, every agent
    receives the flattened `(n_agents, n_subgoals)` matrix (agent by agent), otherwise its own row.

    @ai-generated
    """

    def __init__(self, shaping: LaserCrossingShaping, joint: bool):
        n_agents = shaping.reached.shape[0]
        meanings = shaping.meanings
        if joint:
            meanings = [f"Agent {agent} - {meaning}" for agent in range(n_agents) for meaning in meanings]
        super().__init__(len(meanings), meanings)
        self.shaping = shaping
        self.joint = joint

    def compute(self) -> npt.NDArray[np.float32]:
        reached = self.shaping.reached.astype(np.float32)
        if self.joint:
            return np.tile(reached.reshape(-1), (reached.shape[0], 1))
        return reached
