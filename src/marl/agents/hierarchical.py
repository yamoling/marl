from typing import Literal
import numpy as np
from marlenv import Observation
from marl.models import NN
from torch import device
import torch
from .agent import Agent


class Hierarchical(Agent):
    def __init__(self, manager: NN, workers: Agent):
        super().__init__()
        self.meta = manager
        self.workers = workers

    def choose_action(self, observation: Observation):
        obs_data = torch.from_numpy(observation.data).to(self.device, non_blocking=True)
        with torch.no_grad():
            subgoals = self.meta.forward(obs_data).numpy(force=True)
        observation.extras = np.concat([observation.extras, subgoals], axis=-1)
        return subgoals, self.workers.choose_action(observation)

    def to(self, device: device):
        self.device = device
        self.meta.to(device)
        self.workers.to(device)
        return self

    def randomize(self, method: Literal["xavier"] | Literal["orthogonal"] = "xavier"):
        self.meta.randomize(method)
        self.workers.randomize(method)
