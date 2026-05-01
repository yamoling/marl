from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import torch
from marlenv.models import Observation

from marl.models import Action, Agent

if TYPE_CHECKING:
    from marl.models import Actor


class SimpleAgent[T: torch.distributions.Distribution, U: np.ndarray](Agent[U]):
    def __init__(self, actor_network: "Actor[T]"):
        super().__init__()
        self.actor_network = actor_network

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        with torch.no_grad():
            obs_data, obs_extras, available_actions = observation.as_tensors(self._device, batch_dim=True, actions=True)
            distribution = self.actor_network.policy(obs_data, obs_extras, available_actions)
        actions = distribution.sample().squeeze(0).numpy(force=True)
        actions = cast(U, actions)
        if with_details:
            all_actions = (
                torch.arange(observation.available_actions.shape[-1], device=self._device)
                .repeat_interleave(observation.n_agents)
                .view(-1, observation.n_agents)
            )
            action_probs = distribution.log_prob(all_actions).exp().T
            return Action(actions, action_probabilities=action_probs.numpy(force=True))
        return Action(actions)


class DiscreteAgent(SimpleAgent[torch.distributions.Categorical, npt.NDArray[np.int64]]):
    pass


class DiscreteOneHotAgent(SimpleAgent[torch.distributions.OneHotCategorical, npt.NDArray[np.int64]]):
    pass


@dataclass
class ContinuousAgent(SimpleAgent[torch.distributions.MultivariateNormal, npt.NDArray[np.float32]]):
    pass
