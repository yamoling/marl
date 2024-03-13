import torch

from marl.models.nn import Mixer


class VDN(Mixer):
    def forward(self, qvalues: torch.Tensor, dim: int, *_args, **_kwargs) -> torch.Tensor:
        return qvalues.sum(dim=dim)

    def save(self, directory: str):
        return

    def load(self, directory: str):
        return
