import random
from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from .nn import NN


@dataclass
class OptionCritic(NN):
    n_options: int
    n_agents: int
    options: list[int]

    def __init__(self, n_options: int, n_actions: int, n_agents: int):
        super().__init__(n_actions)
        self.n_options = n_options
        self.n_agents = n_agents
        self.options = [random.randint(0, n_options - 1) for _ in range(n_agents)]

    @abstractmethod
    def compute_q_options(self, obs: Tensor, extras: Tensor) -> Tensor:
        """Compute the Q-values of the options for the given observations."""

    @abstractmethod
    def compute_termination_probability(self, obs: Tensor, extras: Tensor, options: Tensor) -> Tensor:
        """Compute the termination probability of the given aent-wise options"""

    @abstractmethod
    def policy(self, obs: Tensor, extras: Tensor, available_actions: torch.Tensor, options: Tensor) -> torch.distributions.Categorical:
        """Compute the policy distribution for the given observation, extras and following the given options."""
