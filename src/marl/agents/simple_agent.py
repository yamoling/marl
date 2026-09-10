from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from marlenv.models import Observation

from marl.models import Action, Agent
from marl.utils import PinnedStagingBuffer

if TYPE_CHECKING:
    from marl.models import Actor


class SimpleAgent[T: torch.distributions.Distribution](Agent):
    def __init__(self, actor: Actor[T], record_probabilities: bool = False):
        super().__init__()
        self.actor = actor
        self.record_probabilities = record_probabilities
        """Whether to always return the probabilities of the behaviour policy while training. Off-policy
        actor-critic algorithms such as ACER need them to compute their importance sampling weights."""
        self._data_stager = PinnedStagingBuffer()
        self._extras_stager = PinnedStagingBuffer()
        self._available_actions_stager = PinnedStagingBuffer()

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        """
        Select an action from the observation.

        On CUDA, the observation fields are staged through reusable pinned host buffers
        (`PinnedStagingBuffer`) and transferred with non-blocking copies instead of the default
        per-field pageable `Observation.as_tensors` transfer. CPU behaviour is unchanged.

        When `record_probabilities` is set, the details (and thus the probabilities of the behaviour
        policy) are always computed while training, such that they are stored in the transitions.

        @ai-generated
        """
        with_details = with_details or (self.record_probabilities and self.is_training)
        with torch.no_grad():
            if self._device.type == "cuda":
                obs_data = self._data_stager.to(observation.data, self._device).unsqueeze(0)
                obs_extras = self._extras_stager.to(observation.extras, self._device).unsqueeze(0)
                available_actions = self._available_actions_stager.to(
                    observation.available_actions, self._device
                ).unsqueeze(0)
            else:
                obs_data, obs_extras, available_actions = observation.as_tensors(
                    self._device, batch_dim=True, actions=True
                )
            distribution = self.actor.policy(obs_data, obs_extras, available_actions=available_actions)
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


DiscreteAgent = SimpleAgent[torch.distributions.Categorical]
DiscreteOneHotAgent = SimpleAgent[torch.distributions.OneHotCategorical]
ContinuousAgent = SimpleAgent[torch.distributions.MultivariateNormal]
