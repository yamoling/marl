import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from marlenv import Observation

from marl.models import Action, Agent

if TYPE_CHECKING:
    from marl.models import Policy
    from marl.models.nn.options import OptionCriticNetwork


@dataclass
class OptionAgent(Agent):
    """Agent wrapper for Option-Critic policies.

    The trainer and the agent share the same `SimpleOptionCritic` module,
    including the mutable `options` state.
    """

    oc: OptionCriticNetwork
    n_options: int
    n_agents: int
    train_policy: Policy
    test_policy: Policy

    def __init__(
        self, n_options: int, n_agents: int, option_critic: OptionCriticNetwork, train_policy: Policy, test_policy: Policy | None = None
    ):
        Agent.__init__(self)
        self.n_options = n_options
        self.n_agents = n_agents
        self.options = [random.randint(0, n_options - 1) for _ in range(self.n_agents)]
        self.oc = option_critic
        self.policy = train_policy
        self.train_policy = train_policy
        self.test_policy = test_policy if test_policy is not None else train_policy
        self._saved_options = self.options
        self.force_update_next_option = False

    def update_options(self, obs: torch.Tensor, extras: torch.Tensor):
        """Checks for each option if it should terminate or not."""
        # Add the batch dimension for the option (hence the list)
        options = torch.tensor([self.options], dtype=torch.long, device=self.device)
        end_probs = self.oc.termination_probability(obs, extras, options).flatten()
        end_probs = end_probs.numpy(force=True)
        r = np.random.random(end_probs.shape).astype(dtype=np.float32)
        q_options = self.oc.compute_q_options(obs, extras)
        q_options = q_options.view(self.n_agents, self.n_options).numpy(force=True)
        chosen_options = self.policy.get_action(q_options)
        if self.force_update_next_option:
            self.options = chosen_options.tolist()
        else:
            self.options = np.where((r < end_probs), chosen_options, self.options).tolist()
        self.force_update_next_option = False

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        with torch.no_grad():
            obs, extras, available = observation.as_tensors(self.device, batch_dim=True, actions=True)
            self.update_options(obs, extras)
            dist = self.oc.policy(obs, extras, available, self.options)
            action = dist.sample().squeeze(0)
        if with_details:
            return Action(action.numpy(force=True), options=self.options, action_probabilities=dist.probs.numpy(force=True).squeeze(0))
        return Action(action.numpy(force=True))

    def new_episode(self):
        # Upon new episodes, we want to force the agent to update its option at the first step, even if the termination probability is low.
        self.force_update_next_option = True
        return super().new_episode()

    def set_training(self):
        self.policy = self.train_policy
        self.options = self._saved_options
        return super().set_training()

    def set_testing(self):
        self._saved_options = self.options
        self.policy = self.test_policy
        return super().set_testing()
