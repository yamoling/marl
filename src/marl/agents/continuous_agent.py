from dataclasses import dataclass

import torch
from marlenv import Observation
from marl.models.nn import ContinuousActorNN
from typing import Literal

from .agent import Agent


@dataclass
class ContinuousAgent(Agent):
    actor_network: ContinuousActorNN
    distribution_type: Literal["normal", "multivariate_normal"] = "normal"

    def __init__(self, actor_network: ContinuousActorNN, distribution_type: Literal["normal", "multivariate_normal"] = "normal"):
        super().__init__()
        self.actor_network = actor_network
        self.distribution_type = distribution_type

    def choose_action(self, observation: Observation):
        with torch.no_grad():
            obs_data = torch.from_numpy(observation.data).to(self.device, non_blocking=True)
            obs_extras = torch.from_numpy(observation.extras).to(self.device, non_blocking=True)
            # There is no such thing as available actions in continuous action space
            means, stds = self.actor_network.policy(obs_data, obs_extras)
        match self.distribution_type:
            case "normal":
                distribution = torch.distributions.Normal(means, stds)
            case "multivariate_normal":
                distribution = torch.distributions.MultivariateNormal(means, stds)
        return distribution.sample().numpy(force=True)
