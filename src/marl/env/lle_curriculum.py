import random
from numpy import ndarray
from rlenv.wrappers import RLEnvWrapper
from lle import LLE, LaserSource, WorldState
from dataclasses import dataclass
from serde import serde


class CurriculumLearning(RLEnvWrapper):
    def __init__(self, env: LLE, initial_states: list[WorldState], interval: int):
        super().__init__(env)
        self.world = env.world
        self.lle = env
        self.initial_states = initial_states
        self.interval = interval
        self.next_threshold = interval
        self.t = 0
        self.index = 0

    def step(self, action):
        self.t += 1
        return super().step(action)

    def reset(self):
        self.lle.reset()
        while self.t >= self.next_threshold:
            self.next_threshold += self.interval
            if self.index < len(self.initial_states) - 1:
                self.index += 1
        self.world.set_state(self.initial_states[self.index])
        return self.lle.get_observation()


@serde
@dataclass
class RandomInitialStates(RLEnvWrapper):
    accumulate: bool

    def __init__(self, env: LLE, accumulate: bool):
        super().__init__(env)
        self.lle = env
        self.world = env.world
        self.accumulate = accumulate
        self.t = 0
        self.area0 = list(
            set((i, j) for i in range(7, self.world.height) for j in range(7))
            .union((i, j) for i in range(9, self.world.height) for j in range(7, self.world.width))
            .difference(pos for (pos, gem) in self.world.gems)
            .difference(pos for pos in self.world.exit_pos)
        )
        self.area1 = [(5, j) for j in range(self.world.width)]
        self.area2 = [(i, j) for i in range(4) for j in range(2, self.world.width)]

    def get_initial_state(self):
        initial_pos = []
        for agent in range(self.n_agents):
            agent_pos = None
            while agent_pos is None or agent_pos in initial_pos:
                if self.t < 300_000:
                    area = self.area0
                elif self.t < 600_000:
                    if agent == 1 or not self.accumulate:
                        area = self.area1
                    else:
                        area = self.area0 + self.area1
                else:
                    if agent == 0:
                        area = self.area2
                    elif agent == 1:
                        area = self.area1
                    else:
                        if self.accumulate:
                            area = self.area0 + self.area1 + self.area2
                        else:
                            area = self.area2

                agent_pos = random.choice(area)
            initial_pos.append(agent_pos)
        return WorldState(initial_pos, [False] * self.world.n_gems)

    def step(self, action):
        self.t += 1
        return super().step(action)

    def reset(self):
        self.lle.reset()
        initial_state = self.get_initial_state()
        self.lle.set_state(initial_state)
        return self.lle.get_observation()

    def seed(self, seed_value: int):
        random.seed(seed_value)
        return super().seed(seed_value)


@serde
@dataclass
class LaserCurriculum(RLEnvWrapper):
    def __init__(self, env: LLE):
        super().__init__(env)
        self.world = env.world
        self.top_left_laser = self.world.laser_sources[0, 1]
        self.top_laser = self.world.laser_sources[4, 0]
        self.bot_laser = self.world.laser_sources[6, 12]
        self.t = 0

    def randomize(self, source: LaserSource, p_enabled: float, p_colour: float):
        if random.random() <= p_enabled:
            self.world.enable_laser_source(source)
        else:
            self.world.disable_laser_source(source)
        if random.random() <= p_colour:
            colour = random.randint(0, self.n_agents - 1)
            self.world.set_laser_colour(source, colour)

    def step(self, actions: list[int] | ndarray):
        self.t += 1
        return super().step(actions)

    def reset(self):
        """
        - < 100k: disable all lasers
        - < 200k: enable bottom laser with 50% probability
        - < 300k: enable bottom laser with 50% probability with a random colour
        - < 400k: enable bot + top laser with 50% probability. Top laser has a random colour.
        - < 500k: enable bot + top laser with 50% probability. Both lasers have a random colour.
        - < 600k: enable all lasers with random colours.
        - < 700k: enable all lasers with original colours.
        """
        if self.t < 100_000:
            self.world.disable_laser_source(self.top_left_laser)
            self.world.disable_laser_source(self.top_laser)
            self.world.disable_laser_source(self.bot_laser)
        elif self.t < 200_000:
            self.randomize(self.bot_laser, 0.5, 0.0)
        elif self.t < 300_000:
            self.randomize(self.bot_laser, 0.5, 1.0)
        elif self.t < 400_000:
            self.randomize(self.bot_laser, 0.5, 1.0)
            self.randomize(self.top_laser, 0.5, 0.0)
        elif self.t < 500_000:
            self.randomize(self.bot_laser, 0.5, 1.0)
            self.randomize(self.top_laser, 0.5, 1.0)
        elif self.t < 600_000:
            self.randomize(self.bot_laser, 1.0, 1.0)
            self.randomize(self.top_laser, 1.0, 1.0)
            self.randomize(self.top_left_laser, 1.0, 1.0)
        elif self.t < 700_000:
            self.world.enable_laser_source(self.top_left_laser)
            self.world.enable_laser_source(self.bot_laser)
            self.world.enable_laser_source(self.top_laser)
            self.world.set_laser_colour(self.top_laser, 0)
            self.world.set_laser_colour(self.bot_laser, 1)
            self.world.set_laser_colour(self.top_left_laser, 2)

        return super().reset()
