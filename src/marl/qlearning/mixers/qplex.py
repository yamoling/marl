import torch
from dataclasses import dataclass
from serde import serde
from .mixer import Mixer


@serde
@dataclass
class Qplex(Mixer):
    """Duplex dueling"""

    def __init__(self, n_agents: int):
        super().__init__(n_agents)

    def forward(self, qvalues: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        values = torch.max(qvalues, dim=-1, keepdim=True).values
        advantages = qvalues - values
        # Transform the values and the advantages

        q_tot = values + advantages
