from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .nn import NN


@dataclass
class OptionCriticNetwork(NN):
    n_options: int

    def __init__(self, n_options: int):
        super().__init__()
        self.n_options = n_options

    @abstractmethod
    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        """Compute the Q-values of the options for the given observations."""

    @abstractmethod
    def termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        """Compute the termination probability of the given agent-wise options"""

    def value_on_arrival(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        r"""
        Compute the value upon arrival at the given observation and considering the active options as:

        $$U_\Omega(\omega, s') = (1 - \beta_\omega(s')) Q_\Omega(s', \omega) + \beta_\omega(s) V_\Omega(s')$$
        """
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
        options: Sequence[int] | torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Compute the policy distribution for the given observation, extras and following the given options."""
