from abc import abstractmethod
from typing import Optional
import numpy as np

from lle import World, Position, LLE
from lle.tiles import Direction, LaserSource
from marlenv.wrappers import RLEnvWrapper, MARLEnv
from marlenv import Observation, ActionSpace

HORIZONTAL = [Direction.EAST, Direction.WEST]
VERTICAL = [Direction.NORTH, Direction.SOUTH]


class PotentialShaping[A, AS: ActionSpace](RLEnvWrapper[A, AS]):
    """
    Potential shaping for the Laser Learning Environment (LLE).

    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    """

    gamma: float

    def __init__(self, env: MARLEnv[A, AS], gamma: float, extra_shape: Optional[tuple[int]]):
        if extra_shape is None:
            extras_meanings = None
        else:
            extras_meanings = [f"potential-{i}" for i in range(extra_shape[0])]
        super().__init__(env, extra_shape=extra_shape, extra_meanings=extras_meanings)
        self.gamma = gamma

    def add_extras(self, obs: Observation) -> Observation:
        """Add the extras related to potential shaping. Does nothing by default."""
        return obs

    def reset(self):
        obs, state = super().reset()
        return self.add_extras(obs), state

    def step(self, actions: A):
        phi_t = self.current_potential
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        step = super().step(actions)

        self.current_potential = self.compute_potential()
        shaped_reward = self.gamma * phi_t - self.current_potential
        step.obs = self.add_extras(step.obs)
        step.reward += shaped_reward
        return step

    @abstractmethod
    def compute_potential(self) -> float:
        """Compute the potential of the current state of the environment."""


class RewardedPositionSet:
    positions: set[Position]
    already_rewarded: list[bool]
    n_agents: int

    def __init__(self, positions: set[Position], n_agents: int):
        self.positions = positions
        self.already_rewarded = [False] * n_agents
        self.n_agents = n_agents

    def __contains__(self, pos: Position):
        return pos in self.positions

    def reset(self):
        self.already_rewarded = [False] * self.n_agents

    def must_be_rewarded(self, agent_num: int, position: Position):
        if self.already_rewarded[agent_num]:
            return False
        if position in self:
            self.already_rewarded[agent_num] = True
            return True
        return False


class LLEPotentialShaping(PotentialShaping):
    def __init__(
        self,
        env: LLE,
        lasers_to_reward: dict[LaserSource, Direction],
        gamma: float,
        reward_value: float = 1.0,
    ):
        """
        Parameters:
         - env: The environment to wrap.
         - world: The world of the LLE.
         - lasers_to_reward: A dictionary mapping each laser source that has to be rewarded to the direction in which the agents have to move.
         - discount_factor: The discount factor `gamma`
        """
        n_extras = len(lasers_to_reward) * 2
        assert len(env.extra_shape) == 1
        super().__init__(env, gamma, extra_shape=(env.extra_shape[0] + n_extras,))
        self.gamma = gamma
        self.reward_value = reward_value
        self.pos_to_reward = LLEPotentialShaping._compute_positions_to_reward(lasers_to_reward, env.world)
        self.agents_pos_reached = np.full((env.n_agents, len(self.pos_to_reward)), False, dtype=np.bool)
        assert self.agents_pos_reached.shape[1] == n_extras
        self.world = env.world
        self.current_potential = self.compute_potential()

    @staticmethod
    def _compute_positions_to_reward(lasers_to_reward: dict[LaserSource, Direction], world: World):
        pos_to_reward = list[set[Position]]()
        for source, direction in lasers_to_reward.items():
            # Make sure that the source and direction are compatible
            source_is_vertical = source.direction in VERTICAL
            goal_direction_is_horizontal = direction in HORIZONTAL
            assert source_is_vertical == goal_direction_is_horizontal, (
                "The source and direction are incompatible. For horizontal lasers, the direction must be vertical and vice versa."
            )
            in_laser_rewards = set[Position]()
            after_laser_rewards = set[Position]()
            for laser in world.lasers:
                i, j = laser.pos
                if laser.laser_id == source.laser_id:
                    in_laser_rewards.add((i, j))
                    di, dj = direction.delta()
                    ri, rj = (i + di, j + dj)
                    if ri >= 0 and ri < world.width and rj >= 0 and rj < world.height and (ri, rj) not in world.wall_pos:
                        after_laser_rewards.add((i + di, j + dj))
            pos_to_reward.append(in_laser_rewards)
            pos_to_reward.append(after_laser_rewards)
        return pos_to_reward

    def compute_potential(self) -> float:
        for agent_num, agent_pos in enumerate(self.world.agents_positions):
            for j, rewarded_positions in enumerate(self.pos_to_reward):
                if agent_pos in rewarded_positions:
                    self.agents_pos_reached[agent_num, j] = True
        return float(self.agents_pos_reached.size - self.agents_pos_reached.sum()) * self.reward_value

    def add_extras(self, obs: Observation) -> Observation:
        # Each agent is given its own set of reached positions
        extras = self.agents_pos_reached.astype(np.float32)
        obs.extras = np.concatenate([obs.extras, extras], axis=1)
        return super().add_extras(obs)

    def reset(self):
        self.agents_pos_reached.fill(False)
        return super().reset()
