import numpy as np
import numpy.typing as npt
from marlenv import MARLEnv, Observation, Space

from marl.models import Action, Agent


class RandomAgent[S: Space[np.ndarray]](Agent[np.ndarray]):
    def __init__(self, env: MARLEnv[S]):
        super().__init__()
        self.env = env

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        return Action(self.env.action_space.sample(observation.available_actions))

    def value(self, _):
        return 0.0

    def to(self, *args, **kwargs):
        return self

    def set_training(self):
        return

    def set_testing(self):
        return

    def save(self, *args, **kwargs):
        return

    def load(self, *args, **kwargs):
        return

    def randomize(self, *args, **kwargs):
        return


class RandomOneHot(Agent[npt.NDArray[np.int64]]):
    def __init__(self, n_actions: int, n_agents: int):
        self.n_actions = n_actions
        self.n_agents = n_agents

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        zeros = np.zeros((self.n_agents, self.n_actions), dtype=np.int64)
        actions = np.random.randint(0, self.n_actions, size=self.n_agents)
        zeros[np.arange(self.n_agents), actions] = 1
        return Action(zeros)
