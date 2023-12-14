import torch

from .mixer import Mixer


class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, *_) -> torch.Tensor:
        return qvalues.sum(dim=-1, keepdim=True)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return
