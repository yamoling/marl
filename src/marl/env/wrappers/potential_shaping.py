from copy import deepcopy
import numpy as np

from lle import World, Direction, Position
from rlenv import Observation
from rlenv.wrappers import RLEnvWrapper, RLEnv
from serde import serde

HORIZONTAL = [Direction.EAST, Direction.WEST]
VERTICAL = [Direction.NORTH, Direction.SOUTH]


@serde
class PotentialShaping(RLEnvWrapper):
    """
    Potential shaping for the Laser Learning Environment (LLE).

    https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    """

    def __init__(
        self,
        env: RLEnv,
        world: World,
        lasers_to_reward: dict[Position, Direction],
        discount_factor: float,
        reward_value: float = 1.0,
    ):
        """
        Parameters:
         - env: The environment to wrap.
         - world: The world of the LLE.
         - lasers_to_reward: A dictionary mapping each laser that has to be rewarded to the direction in which the agents have to move.
         - discount_factor: The discount factor `gamma`
         - delay: The delay `d` for the delayed reward.
        """
        n_extras = env.n_agents * len(world.laser_sources) * 2
        assert len(env.extra_feature_shape) == 1
        super().__init__(env, extra_feature_shape=(env.extra_feature_shape[0] + n_extras,))
        self.world = world
        self.gamma = discount_factor
        self.reward_value = reward_value

        self.pos_to_reward = PotentialShaping._compute_positions_to_reward(lasers_to_reward, world)
        self.agents_pos_reached = np.full((env.n_agents, len(self.pos_to_reward)), False, dtype=bool)
        assert self.agents_pos_reached.size == n_extras
        self.current_potential = self.compute_potential()

    @staticmethod
    def _compute_positions_to_reward(lasers_to_reward: dict[Position, Direction], world: World):
        pos_to_reward = list[list[Position]]()
        for source_pos, direction in lasers_to_reward.items():
            # Make sure that the source and direction are compatible
            source = world.laser_sources[source_pos]
            source_is_vertical = source.direction in VERTICAL
            goal_direction_is_horizontal = direction in HORIZONTAL
            assert (
                source_is_vertical == goal_direction_is_horizontal
            ), "The source and direction are incompatible. For horizontal lasers, the direction must be vertical and vice versa."
            in_laser_rewards = list[Position]()
            after_laser_rewards = list[Position]()
            for (i, j), laser in world.lasers:
                if laser.laser_id == source.laser_id:
                    in_laser_rewards.append((i, j))
                    (di, dj) = direction.delta()
                    after_laser_rewards.append((i + di, j + dj))
            pos_to_reward += [in_laser_rewards, after_laser_rewards]
        return pos_to_reward

    def add_extras(self, obs: Observation) -> Observation:
        extras = self.agents_pos_reached.flatten()
        agents_extras = np.array([extras for _ in range(self.n_agents)], dtype=np.float32)
        obs.extras = np.concatenate([obs.extras, agents_extras], axis=1)
        return obs

    def reset(self):
        self.agents_pos_reached.fill(False)
        obs = super().reset()
        return self.add_extras(obs)

    def step(self, actions: list[int] | np.ndarray):
        phi_t = self.current_potential
        obs, r, done, truncated, info = super().step(actions)

        self._update_potential()
        self.current_potential = self.compute_potential()
        shaped_reward = self.gamma * phi_t - self.current_potential
        return self.add_extras(obs), r + shaped_reward, done, truncated, info

    def _update_potential(self):
        for i, pos in enumerate(self.world.agents_positions):
            for j, rewarded_positions in enumerate(self.pos_to_reward):
                if pos in rewarded_positions:
                    self.agents_pos_reached[i, j] = True

    def compute_potential(self):
        return (self.agents_pos_reached.size - self.agents_pos_reached.sum()) * self.reward_value
