from dataclasses import dataclass

import torch

from marl.models.batch import Batch
from marl.models.nn import Mixer


@dataclass
class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim=self.agent_dim)

    def forward_batch(self, qvalues: torch.Tensor, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """VDN ignores the states, so do not load them from the batch."""
        return self.forward(qvalues)

    def __hash__(self):
        return id(self)
