from dataclasses import dataclass

import torch

from marl.models.nn import StateMixer


@dataclass
class VDN(StateMixer):
    def forward(self, qvalues: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim=self.agent_dim)

    def __hash__(self):
        return id(self)
