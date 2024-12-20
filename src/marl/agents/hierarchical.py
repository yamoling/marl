from typing import Literal
import numpy as np
from marlenv import Observation
from torch import device
from .agent import Agent


class Hierarchical(Agent):
    """
    Hierarchical agent where a meta-agent gives orders (i.e. actions, subgoals, communication) to workers (multi-agent).

    The subgoals are concatenated to the worker's observations as "extras".
    """

    def __init__(self, manager: Agent, workers: Agent):
        super().__init__()
        self.meta = manager
        self.workers = workers

    def choose_action(self, observation: Observation):
        subgoals = self.meta.choose_action(observation)
        observation.extras = np.concat([observation.extras, subgoals], axis=-1)
        workers_actions = self.workers.choose_action(observation)
        return subgoals, workers_actions

    def to(self, device: device):
        self.device = device
        self.meta.to(device)
        self.workers.to(device)
        return self

    def randomize(self, method: Literal["xavier"] | Literal["orthogonal"] = "xavier"):
        self.meta.randomize(method)
        self.workers.randomize(method)
