from abc import ABC, abstractmethod
from dataclasses import dataclass
from serde import serde
from typing import Literal
from typing_extensions import Self
from rlenv import Transition, Episode

import torch

@serde
@dataclass
class Trainer(ABC):
    """Algorithm trainer class. Needed to train an algorithm but not to test it."""
    update_interval: int
    """
    How often to update the algorithm. 
    If the algorithm is trained on episodes, this is the number of episodes between each update.
    If the algorithm is trained on steps, this is the number of steps between each update.
    """
    update_on_steps: bool
    """Whether to update on steps."""
    update_on_episodes: bool
    """Whether to update on episodes."""

    def __init__(self, update_type: Literal["step", "episode"], update_interval: int):
        assert update_type in ["step", "episode"]
        assert update_interval > 0
        self.update_after_each = update_type
        self.update_interval = update_interval
        self.update_on_steps = update_type == "step"
        self.update_on_episodes = update_type == "episode"


    def update_step(self, transition: Transition, time_step: int):
        """Update to call after each step. Should be run when update_after_each == "step"."""

    def update_episode(self, episode: Episode, time_step: int):
        """Update to call after each episode. Should be run when update_after_each == "episode"."""

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        """Send the tensors to the given device."""

    @abstractmethod
    def save(self, to_directory: str):
        """Save the trainer to the given directory."""

    @abstractmethod
    def load(self, from_directory: str):
        """Load the trainer from the given directory."""

    @abstractmethod
    def randomize(self):
        """Randomize the state of the trainer."""