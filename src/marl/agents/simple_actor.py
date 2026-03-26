from dataclasses import dataclass

import torch
from marlenv.models import Observation

from marl.models import Actor, Agent


@dataclass
class SimpleActor(Agent):
    actor_network: Actor

    def __init__(self, actor_network: Actor):
        super().__init__()
        self.actor_network = actor_network

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            obs_data, obs_extras = observation.as_tensors(self._device, batch_dim=True)
            available_actions = torch.from_numpy(observation.available_actions).unsqueeze(0).to(self._device, non_blocking=True)
            distribution = self.actor_network.policy(obs_data, obs_extras, available_actions)
        actions = distribution.sample().squeeze(0)
        return actions.numpy(force=True)
