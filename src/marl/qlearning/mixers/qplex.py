import torch
from dataclasses import dataclass
from serde import serde
from marl.models.nn import Mixer


@serde
@dataclass
class QPlex(Mixer):
    """Duplex dueling"""

    def __init__(self, n_agents: int):
        super().__init__(n_agents)

    def _transformation(self, qvalues: torch.Tensor, states: torch.Tensor):
        """First step described in the paper is called 'transformation'"""
        w, b = self._get_weights_and_biases(states)
        qvalues = qvalues * w + b
        values = torch.max(qvalues, dim=-1, keepdim=True).values
        advantages = qvalues - values
        return values, advantages

    def _duelling_mixing(self, values: torch.Tensor, advantages: torch.Tensor, states: torch.Tensor):
        lambdas = self._get_lambdas(states)
        values = torch.sum(values, dim=-1)
        total_advantages = torch.dot(advantages, lambdas)
        return values + total_advantages

    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        values, advantages = self._transformation(qvalues, states)
        return self._duelling_mixing(values, advantages, states)
