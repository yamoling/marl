from abc import ABC
from dataclasses import dataclass, asdict
import os
from typing import Any, Optional

import torch
from marlenv import Episode, Transition, Observation, State

from marl.agents import Agent
from marl.utils import HasDevice


@dataclass
class Trainer(HasDevice, ABC):
    """Algorithm trainer class. Needed to train an algorithm but not to test it."""

    name: str

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.name = self.__class__.__name__

    def make_agent(self) -> Agent:
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

    def value(self, obs: Observation, state: State) -> float:
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
