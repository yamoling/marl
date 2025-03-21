from abc import ABC
from dataclasses import dataclass
from typing import Literal, Any, Optional
from typing_extensions import Self
from marlenv import Transition, Episode
from marl.models import NN, Batch
from marl.agents import Agent
from .nn import IRModule

import torch


@dataclass
class Trainer[A](ABC):
    """Algorithm trainer class. Needed to train an algorithm but not to test it."""

    name: str
    update_on_steps: bool
    """Whether to update on steps."""
    update_on_episodes: bool
    """Whether to update on episodes."""

    def __init__(self, update_type: Literal["step", "episode", "both"] = "both"):
        self.name = self.__class__.__name__
        self.update_on_steps = update_type in ["step", "both"]
        self.update_on_episodes = update_type in ["episode", "both"]
        self._device = torch.device("cpu")

    def make_agent(self, *, ir_module: Optional[IRModule] = None) -> Agent:
        raise NotImplementedError("Trainer must implement make_agent method")

    def update_step(self, transition: Transition[A], time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def update_episode(self, episode: Episode[A], episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def values(self, batch: Batch) -> torch.Tensor:
        """Compute the value of the batch."""
        raise NotImplementedError("Trainer did not implement the value method")

    def next_values(self, batch: Batch) -> torch.Tensor:
        """Compute the value of the next batch."""
        raise NotImplementedError("Trainer did not implement the next_values method")

    def to(self, device: torch.device) -> Self:
        """Send the networks to the given device."""
        self._device = device
        for nn in self.networks:
            nn.to(device)
        return self

    @property
    def networks(self):
        """Dynamic list of neural networks attributes in the trainer"""
        return [nn for nn in self.__dict__.values() if isinstance(nn, NN)]

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the state of the trainer."""
        for nn in self.networks:
            nn.randomize(method)
