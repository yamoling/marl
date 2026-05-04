import torch

from marl.config import VDNConfig
from marl.models.nn import Mixer


class VDN(Mixer, VDNConfig):
    def forward(self, qvalues: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Sum across the agent dimension
        return torch.sum(qvalues, dim=self.agent_dim)

    def save(self, to_directory: str):
        return

    def load(self, from_directory: str):
        return
