from ..agent import Agent

from typing import Literal
import numpy as np
import torch
from marl.models import NN
from marlenv import Observation
from torch import device

from dataclasses import dataclass


@dataclass
class Haven(Agent):
    """
    Hierarchical agent where a meta-agent gives orders (i.e. actions, subgoals, communication) to workers (multi-agent).

    The subgoals are concatenated to the worker's observations as "extras".
    """

    manager: NN
    workers: Agent
    n_subgoals: int
    k: int
    n_agents: int

    def __init__(self, manager: NN, workers: Agent, n_subgoals: int, k: int):
        super().__init__()
        self.meta = manager
        self.workers = workers
        self.k = k
        self.meta_extras_len = manager.extras_shape[0]
        self.n_subgoals = n_subgoals
        self.n_agents = manager.output_shape[0]
        self.indices = np.arange(self.n_agents)

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            obs_data = torch.from_numpy(observation.data[0]).unsqueeze(0).to(self.device)
            extras = torch.from_numpy(observation.extras[0][: self.meta_extras_len]).unsqueeze(0).to(self.device)
            subgoals = self.meta.forward(obs_data, extras).squeeze(0).numpy(force=True)
        observation.extras[self.indices, -self.n_subgoals :] = subgoals
        workers_actions = self.workers.choose_action(observation)
        return subgoals, workers_actions

    def new_episode(self):
        super().new_episode()
        self.workers.new_episode()

    def to(self, device: device):
        self.device = device
        self.meta.to(device)
        self.workers.to(device)
        return self

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta.randomize(method)
        self.workers.randomize(method)
