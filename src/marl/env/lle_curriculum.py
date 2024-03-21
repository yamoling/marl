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
    def __init__(self, env: LLE, states: dict[int, list[WorldState]]):
        super().__init__(env)
        self.world = env.world
        self.lle = env
        self.change_steps = list(states.keys())
        self.change_steps.sort()
        self.states = [states[step] for step in self.change_steps]
        self.change_steps = self.change_steps[1:] + [float("inf")]
        self.t = 0
        self.current_set = 0

    def step(self, action):
        self.t += 1
        print(self.world.get_state())
        return super().step(action)

    def reset(self):
        if self.t >= self.change_steps[self.current_set]:
            self.current_set += 1
        initial_states_set = self.states[self.current_set]
        initial_state = random.choice(initial_states_set)
        self.world.set_state(initial_state)
        return self.lle.get_observation()
