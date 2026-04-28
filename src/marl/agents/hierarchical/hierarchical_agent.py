from typing import Literal

import torch

from marl.models import Agent


class HierarchicalAgent[T, U](Agent[T]):
    def __init__(self, meta_agent: Agent[U], workers: Agent[T]):
        super().__init__()
        self.meta_agent = meta_agent
        self.workers = workers

    def new_episode(self):
        self.workers.new_episode()
        self.meta_agent.new_episode()
        self._t = 0

    def set_testing(self):
        self.meta_agent.set_testing()
        self.workers.set_testing()
        # Save the train time step and restore the test time step
        self._t_train = self._t
        self._t = self._t_test

    def set_training(self):
        self.meta_agent.set_training()
        self.workers.set_training()
        # Save the test time step and restore the train time step
        self._t_test = self._t
        self._t = self._t_train

    def to(self, device: torch.device):
        self.meta_agent.to(device)
        self.workers.to(device)
        return super().to(device)

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        self.meta_agent.randomize(method)
        self.workers.randomize(method)
