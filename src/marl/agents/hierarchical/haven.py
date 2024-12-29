from dataclasses import dataclass
from copy import copy
from typing import Literal

import numpy as np
from marlenv import Observation
from torch import device

from ..agent import Agent


@dataclass
class Haven(Agent[tuple[np.ndarray, dict[str, np.ndarray]]]):
    """
    Hierarchical agent where a meta-agent gives orders (i.e. actions, subgoals, communication) to workers (multi-agent).

    The subgoals are concatenated to the worker's observations as "extras".
    """

    meta: Agent[np.ndarray]
    workers: Agent[np.ndarray]
    n_subgoals: int
    k: int
    """Number of steps between meta-actions."""

    def __init__(self, meta_agent: Agent, workers: Agent[np.ndarray], n_subgoals: int, k: int, n_meta_extras: int, n_agent_extras: int):
        super().__init__()
        self.meta = meta_agent
        self.workers = workers
        self.k = k
        self.n_subgoals = n_subgoals
        self.t = 0
        self.last_meta_action = np.zeros(0, dtype=np.float32)
        self.n_meta_extras = n_meta_extras
        self.n_agent_extras = n_agent_extras

    def choose_action(self, observation: Observation):
        assert observation.extras_shape[0] == self.n_meta_extras + self.n_agent_extras + self.n_subgoals
        if self.t % self.k == 0:
            meta_obs = copy(observation)
            meta_obs.data = np.expand_dims(observation.data[0], 0)
            # remove the subgoals padding
            meta_obs.extras = np.expand_dims(observation.extras[0, : self.n_meta_extras], 0)
            self.last_meta_action = self.meta.choose_action(meta_obs)
        self.t += 1
        observation.extras[:, -self.n_subgoals :] = self.last_meta_action.squeeze(0)
        workers_actions = self.workers.choose_action(observation)
        return workers_actions, {"meta_actions": self.last_meta_action}

    def new_episode(self):
        super().new_episode()
        self.workers.new_episode()
        self.meta.new_episode()
        self.t = 0

    def to(self, device: device):
        self.device = device
        self.meta.to(device)
        self.workers.to(device)
        return self

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta.randomize(method)
        self.workers.randomize(method)
