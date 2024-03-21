import random
from rlenv.wrappers import RLEnvWrapper
from lle import LLE, WorldState


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


class RandomInitialStates(RLEnvWrapper):
    def __init__(self, env: LLE, accumulate: bool = False):
        super().__init__(env)
        self.lle = env
        self.world = env.world
        self.t = 0
        self.area0 = [(i, j) for i in range(7, self.world.height) for j in range(7)]
        self.area0 += [(i, j) for i in range(9, self.world.height) for j in range(7, self.world.width) if (i, j) not in self.world.exit_pos]
        self.area1 = [(5, j) for j in range(self.world.width)]
        if accumulate:
            self.area1 += self.area0
        self.area2 = [(i, j) for i in range(4) for j in range(2, self.world.width)]
        if accumulate:
            self.area2 = self.area2 + self.area1

    def get_initial_state(self):
        if self.t < 300_000:
            area = self.area0
        elif self.t < 600_000:
            area = self.area1
        else:
            area = self.area2
        pos = random.sample(area, k=self.world.n_agents)
        return WorldState(pos, [False] * self.world.n_gems)

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
