from abc import ABC
from typing import Literal, Optional, Self

import torch


class HasDevice(ABC):
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        self._device = device

    @property
    def networks(self):
        """Dynamic list of neural networks attributes in the trainer"""
        from marl.models.nn import NN

        return [nn for nn in self.__dict__.values() if isinstance(nn, NN)]

    @property
    def device(self):
        return self._device

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the state of the trainer."""
        for nn in self.networks:
            nn.randomize(method)

    def to(self, device: torch.device) -> Self:
        """Send the networks to the given device."""
        self._device = device
        for nn in self.networks:
            nn.to(device)
        return self
