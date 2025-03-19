from dataclasses import dataclass

import torch
from marlenv import Observation
from marl.models.nn import ActorCriticNN
from .agent import Agent


@dataclass
class Actor(Agent):
    actor_network: ActorCriticNN

    def __init__(self, actor_network: ActorCriticNN):
        super().__init__()
        self.actor_network = actor_network

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            obs_data = torch.from_numpy(observation.data).unsqueeze(0).to(self._device, non_blocking=True)
            obs_extras = torch.from_numpy(observation.extras).unsqueeze(0).to(self._device, non_blocking=True)
            available_actions = torch.from_numpy(observation.available_actions).unsqueeze(0).to(self._device, non_blocking=True)
            distribution = self.actor_network.policy(obs_data, obs_extras, available_actions)
        actions = distribution.sample().squeeze(0)
        return actions.numpy(force=True)
