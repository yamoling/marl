from abc import ABC, abstractmethod

from marl.models import Trainer
from rlenv import Transition, Episode
from marl.qlearning import CNetAlgo

from dataclasses import dataclass
from serde import serialize
from typing import Literal, Any
from typing_extensions import Self

import torch


@serialize
@dataclass
class CNetTrainer(Trainer):
    def __init__(self, opt,  agents, update_type: Literal["step", "episode"], update_interval: int):
        super().__init__(update_type, update_interval)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        
        Get the infos from the agents
        """
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        # Iterate on every agent
        
        return {}


    def to(self, device: torch.device) -> Self:
        """Send the tensors to the given device."""
        pass

    def randomize(self):
        """Randomize the state of the trainer."""
        pass