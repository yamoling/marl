from dataclasses import dataclass
from copy import copy
from typing import Literal

import numpy as np
import torch
from marlenv import Observation
from torch import device

from ..agent import Agent


@dataclass
class Haven(Agent):
    """
    Hierarchical agent where a meta-agent gives orders (i.e. actions, subgoals, communication) to workers (multi-agent).

    The subgoals are concatenated to the worker's observations as "extras".
    """

    meta: Agent
    workers: Agent
    n_subgoals: int
    k: int
    """Number of steps between meta-actions."""

    def __init__(self, meta_agent: Agent, workers: Agent, n_subgoals: int, k: int):
        assert meta_agent.n_agents == 1, "Meta-agent must be a single agent"
        super().__init__(workers.n_agents + 1)
        self.meta = meta_agent
        self.workers = workers
        self.k = k
        self.n_subgoals = n_subgoals
        self.indices = np.arange(workers.n_agents)
        self.t = 0
        self.last_meta_action = np.zeros(0, dtype=np.float32)

    def choose_action(self, observation: Observation):
        if self.t % self.k == 0:
            meta_obs = copy(observation)
            meta_obs.data = np.expand_dims(observation.data[0], 0)
            meta_obs.extras = np.expand_dims(observation.extras[0], 0)
            match self.meta.choose_action(meta_obs):
                case (action, dict(info)):
                    self.last_meta_action = action
                case action:
                    self.last_meta_action = action
                    info = {}
        self.t += 1
        observation.extras[:, -self.n_subgoals :] = self.last_meta_action
        workers_actions = self.workers.choose_action(observation)
        return workers_actions, dict(meta_actions=self.last_meta_action) | info

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
