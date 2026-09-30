from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from marlenv.models import Observation

from marl.models import Action, Agent

if TYPE_CHECKING:
    from marl.models import Actor


class SimpleAgent[T: torch.distributions.Distribution](Agent):
    def __init__(self, actor: Actor[T], record_probabilities: bool = False):
        super().__init__()
        self.actor = actor
        self.record_probabilities = record_probabilities
        """Whether to always return the probabilities of the behaviour policy while training. Off-policy
        actor-critic algorithms such as ACER need them to compute their importance sampling weights."""

    @staticmethod
    def from_actor(actor: Actor, record_probabilities: bool = False) -> SimpleAgent:
        """
        Build the most specific agent for the given actor: a `DiscreteAgent` (greedy at test time)
        for categorical actors, and a plain sampling `SimpleAgent` otherwise.
        """
        from marl.models.nn import CategoricalActor

        if isinstance(actor, CategoricalActor):
            return DiscreteAgent(actor, record_probabilities)
        return SimpleAgent(actor, record_probabilities)

    def _select_actions(self, distribution: T) -> torch.Tensor:
        """Select the actions from the policy distribution. By default, sample from it."""
        return distribution.sample()

    @override
    def choose_action(self, observation: Observation, *, with_details: bool = False):
        """
        Select an action from the observation.

        When `record_probabilities` is set, the details (and thus the probabilities of the behaviour
        policy) are always computed while training, such that they are stored in the transitions.
        """
        with_details = with_details or (self.record_probabilities and self.is_training)
        with torch.no_grad():
            obs_data, obs_extras, available_actions = observation.as_tensors(self._device, batch_dim=True, actions=True)
            distribution = self.actor.policy(obs_data, obs_extras, available_actions=available_actions)
        actions = self._select_actions(distribution).squeeze(0).numpy(force=True)
        if with_details:
            all_actions = (
                torch.arange(observation.available_actions.shape[-1], device=self._device)
                .repeat_interleave(observation.n_agents)
                .view(-1, observation.n_agents)
            )
            action_probs = distribution.log_prob(all_actions).exp().T
            return Action(actions, action_probabilities=action_probs.numpy(force=True))
        return Action(actions)


class DiscreteAgent(SimpleAgent[torch.distributions.Categorical]):
    """Categorical agent that samples while training and acts greedily (argmax) while testing."""

    @override
    def _select_actions(self, distribution: torch.distributions.Categorical) -> torch.Tensor:
        if self.is_training:
            return distribution.sample()
        return distribution.logits.argmax(dim=-1)


class DiscreteOneHotAgent(SimpleAgent[torch.distributions.OneHotCategorical]):
    """One-hot categorical agent that samples while training and acts greedily (argmax) while testing."""

    @override
    def _select_actions(self, distribution: torch.distributions.OneHotCategorical) -> torch.Tensor:
        if self.is_training:
            return distribution.sample()
        greedy = distribution.logits.argmax(dim=-1)
        return torch.nn.functional.one_hot(greedy, distribution.event_shape[-1]).to(distribution.probs.dtype)


ContinuousAgent = SimpleAgent[torch.distributions.MultivariateNormal]
