from typing import Literal

import torch

from .agent import Agent


class HierarchicalAgent[T, U](Agent[T]):
    def __init__(self, workers: Agent[T], meta_agent: Agent[U]):
        super().__init__()
        self.meta_agent = meta_agent
        self.workers = workers

    def new_episode(self):
        self.workers.new_episode()
        self.meta_agent.new_episode()

    def set_testing(self):
        self.meta_agent.set_testing()
        self.workers.set_testing()

    def set_training(self):
        self.meta_agent.set_training()
        self.workers.set_training()

    def to(self, device: torch.device):
        self.meta_agent.to(device)
        self.workers.to(device)
        return super().to(device)

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta_agent.randomize(method)
        self.workers.randomize(method)

    def seed(self, seed: int):
        self.workers.seed(seed)
        self.meta_agent.seed(seed)
