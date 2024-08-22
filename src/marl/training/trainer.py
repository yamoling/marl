from abc import ABC, abstractmethod
from dataclasses import dataclass
from serde import serde
from typing import Literal, Any
from typing_extensions import Self
from marlenv import Transition, Episode

import torch


@serde
@dataclass
class Trainer(ABC):
    """Algorithm trainer class. Needed to train an algorithm but not to test it."""

    name: str
    step_update_interval: int
    episode_update_interval: int
    """
    How often to update the algorithm. 
    If the algorithm is trained on episodes, this is the number of episodes between each update.
    If the algorithm is trained on steps, this is the number of steps between each update.
    """
    update_on_steps: bool
    """Whether to update on steps."""
    update_on_episodes: bool
    """Whether to update on episodes."""

    def __init__(self, update_type: Literal["step", "episode", "both"], update_interval: int | tuple[int, int]):
        assert update_type in ["step", "episode", "both"]
        match update_interval:
            case tuple((interval_steps, interval_episodes)):
                assert update_type == "both"
                assert interval_steps > 0
                self.step_update_interval = interval_steps
                assert interval_episodes > 0
                self.episode_update_interval = interval_episodes
            case int(interval):
                assert interval > 0
                self.step_update_interval = interval
                self.episode_update_interval = interval
            case _:
                raise ValueError("Invalid update interval")
        self.name = self.__class__.__name__
        self.update_on_steps = update_type in ["step", "both"]
        self.update_on_episodes = update_type in ["episode", "both"]

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

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        """Send the tensors to the given device."""

    @abstractmethod
    def randomize(self):
        """Randomize the state of the trainer."""
