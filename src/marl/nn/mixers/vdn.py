from dataclasses import dataclass

import torch

from marl.models.nn import Mixer


@dataclass(unsafe_hash=True)
class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim=self.agent_dim)
