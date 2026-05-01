from dataclasses import dataclass

import torch
from marlenv import Observation

from ..action import Action
from .agent import Agent


@dataclass
class AgentWrapper[T](Agent[T]):
    def __init__(self, agent: Agent[T]):
        super().__init__()
        self.agent = agent

    def choose_action(self, observation: Observation, *, with_details: bool = False) -> Action:
        return self.agent.choose_action(observation, with_details=with_details)

    def new_episode(self):
        return self.agent.new_episode()

    def set_training(self):
        self.agent.set_training()
        super().set_training()

    def set_testing(self):
        self.agent.set_testing()
        super().set_testing()

    def to(self, device: torch.device):
        self.agent.to(device)
        return super().to(device)

    def seed(self, seed: int):
        return self.agent.seed(seed)

    def _can_autosave(self):
        if not self.agent._can_autosave():
            return False
        return super()._can_autosave()

    def networks(self):
        all_networks = self.agent.networks()
        wrapper_networks = super().networks()
        return wrapper_networks + all_networks
