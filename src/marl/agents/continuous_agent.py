from dataclasses import dataclass

import torch
from marlenv import Observation
from marl.models.nn import ContinuousActorNN

from .agent import Agent


@dataclass
class ContinuousAgent(Agent):
    actor_network: ContinuousActorNN

    def __init__(self, actor_network: ContinuousActorNN):
        super().__init__()
        self.actor_network = actor_network

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            obs_data = torch.from_numpy(observation.data).unsqueeze(0).to(self.device, non_blocking=True)
            obs_extras = torch.from_numpy(observation.extras).unsqueeze(0).to(self.device, non_blocking=True)
            # There is no such thing as available actions in continuous action space
            distribution = self.actor_network.policy(obs_data, obs_extras)
        actions = distribution.sample().squeeze(0)
        return actions.numpy(force=True)
