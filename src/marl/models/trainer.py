from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional

import torch
from marlenv import Episode, Transition

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
