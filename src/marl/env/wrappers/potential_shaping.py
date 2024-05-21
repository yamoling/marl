from copy import deepcopy
from lle import World, Direction, Position
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
        delay: int = 0,
    ):
        """
        @param env: The environment to wrap.
        @param world: The world of the LLE.
        @param lasers_to_reward: A dictionary mapping each laser that has to be rewarded to the direction in which the agents have to move.
        @param discount_factor: The discount factor `gamma`
        @param delay: The delay `d` for the delayed reward.
        """
        super().__init__(env)
        self.world = world
        self.delay = delay
        self.delayed_rewards = [0.0] * len(world.agents)
        self.gamma = discount_factor

        pos_to_reward = list[set[Position]]()
        for source_pos, direction in lasers_to_reward.items():
            # Make sure that the source and direction are compatible
            source = world.laser_sources[source_pos]
            print(source.direction in HORIZONTAL)
            print(direction in VERTICAL)
            assert (
                source.direction in HORIZONTAL == direction in VERTICAL
            ), "The source and direction are incompatible. For horizontal lasers, the direction must be vertical and vice versa."
            laser_rewards = set[Position]()
            for (i, j), laser in world.lasers:
                if laser.laser_id == source.laser_id:
                    laser_rewards.add((i, j))
                    (di, dj) = direction.delta()
                    laser_rewards.add((i + di, j + dj))
            pos_to_reward.append(laser_rewards)
        pos_reached = [False for _ in pos_to_reward]

        self.agents_pos_to_reward = [deepcopy(pos_to_reward) for _ in world.agents]
        self.agents_pos_reached = [deepcopy(pos_reached) for _ in world.agents]
