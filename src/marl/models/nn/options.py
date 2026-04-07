from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .nn import NN


@dataclass
class OptionCritic(NN):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        """Compute the Q-values of the options for the given observations."""

    @abstractmethod
    def termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        """Compute the termination probability of the given agent-wise options"""

    def value_on_arrival(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        """Compute the value of the current state or observation."""
        termination_probs = self.termination_probability(obs, extras, options)
        q_options = self.compute_q_options(obs, extras)
        best_q_options = q_options.max(dim=-1).values
        current_q_options = torch.gather(q_options, dim=-1, index=options).squeeze(-1)
        return (1 - termination_probs) * current_q_options + termination_probs * best_q_options

    @abstractmethod
    def policy(
        self,
        obs: Tensor,
        extras: Tensor,
        available_actions: torch.Tensor,
        option: Sequence[int],
    ) -> torch.distributions.Categorical:
        """Compute the policy distribution for the given observation, extras and following the given options."""
