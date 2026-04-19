from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from marlenv.models import Observation

from marl.models import Agent, Action

if TYPE_CHECKING:
    from marl.models import Actor


@dataclass
class SimpleActor(Agent):
    actor_network: "Actor"

    def __init__(self, actor_network: "Actor"):
        super().__init__()
        self.actor_network = actor_network

    def choose_action(self, observation: Observation, *, with_details: bool = False) -> Action:
        with torch.no_grad():
            obs_data, obs_extras = observation.as_tensors(self._device, batch_dim=True)
            available_actions = torch.from_numpy(observation.available_actions).unsqueeze(0).to(self._device, non_blocking=True)
            distribution = self.actor_network.policy(obs_data, obs_extras, available_actions)
        actions = distribution.sample().squeeze(0).numpy(force=True)
        if with_details:
            all_actions = (
                torch.arange(observation.available_actions.shape[-1], device=self._device)
                .repeat_interleave(observation.n_agents)
                .view(-1, observation.n_agents)
            )
            action_probs = distribution.log_prob(all_actions).exp().T
            return Action(actions, action_probabilities=action_probs.numpy(force=True))
        return Action(actions)
