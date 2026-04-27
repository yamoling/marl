import os
from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Self, Sequence

import torch
from marlenv import Episode, Observation, State, Transition

from .agent import Agent
from .nn import NN, randomize


@dataclass
class Trainer[T](ABC):
    """Algorithm trainer class. Needed to train an algorithm but not to test it."""

    name: str = field(init=False)

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        self._device = device
        self.name = self.__class__.__name__

    def make_agent(self) -> Agent[T]:
        raise NotImplementedError("Trainer must implement make_agent method")

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def value(self, obs: Observation, state: State) -> float | Sequence:
        """
        Compute the value of the current state or observation.
        """
        return 0.0

    def config(self) -> dict[str, Any]:
        """
        Get the configuration of the trainer, typically used for logging.
        """
        return asdict(self)

    def save(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for i, nn in enumerate(self.networks):
            os.path.join(directory_path, f"{nn.name}_{i}.pt")
            torch.save(nn.state_dict(), f"{directory_path}_{i}.pt")

    def load(self, directory_path: str):
        for i, nn in enumerate(self.networks):
            path = os.path.join(directory_path, f"{nn.name}_{i}.pt")
            nn.load_state_dict(torch.load(path))

    @property
    def device(self):
        return self._device

    @property
    def networks(self):
        """Dynamic list of neural networks attributes in the trainer"""
        return [nn for nn in self.__dict__.values() if isinstance(nn, (NN, torch.nn.Module))]

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the parameters of all the neural networks in the trainer."""

        for nn in self.networks:
            if isinstance(nn, NN):
                nn.randomize(method)
            else:
                randomize(torch.nn.init.xavier_uniform_, nn)

    def to(self, device: torch.device) -> Self:
        """Send the networks to the given device."""
        self._device = device
        for nn in self.networks:
            nn.to(device)
        return self
