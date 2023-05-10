from typing_extensions import Self
from abc import ABC, abstractmethod
import torch
import numpy as np
from rlenv.models import Episode, Transition, Observation


class RLAlgo(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        """Get the action to perform given the input observation"""

    @property
    def name(self) -> str:
        """The name of the algorithm"""
        return self.__class__.__name__
    
    def summary(self) -> dict:
        """Dictionary of the relevant algorithm parameters for experiment logging purposes"""
        return {
            "name": self.name
        }
    
    def to(self, device: torch.device):
        """Move the algorithm to the specified device"""
        raise NotImplementedError("Not implemented for this algorithm")

    def save(self, to_path: str):
        """Save the algorithm state to the specified file."""
        raise NotImplementedError("Not implemented for this algorithm")

    def load(self, from_path: str):
        """Load the algorithm state from the specified file."""
        raise NotImplementedError("Not implemented for this algorithm")

    @classmethod
    def from_summary(cls, summary: dict) -> Self:
        """Instantiate the algorithm from its summary"""
        raise NotImplementedError(f"From summary not implemented for {cls.__name__}")

    def before_tests(self, time_step: int):
        """Hook before tests, for instance to swap from training to testing policy."""

    def after_tests(self, episodes: list[Episode], time_step: int):
        """
        Hook after tests.
        Subclasses should swap from testing back to training policy.
        """
    def after_train_step(self, transition: Transition, time_step: int):
        """Hook after every training step."""

    def before_train_episode(self, episode_num: int):
        """Hook before every training episode."""

    def after_train_episode(self, episode_num: int, episode: Episode):
        """Hook after every training episode."""

    def before_test_episode(self, time_step: int, test_num: int):
        """
        Hook before each train episode
        
        - time_step: the training step at which the test happens
        - test_num: the test number for this time step
        """

    def after_test_episode(self, time_step: int, test_num: int, episode: Episode):
        """
        Hook after each test episode.

        - time_step: the training step at which the tests occurs
        - test_num: the test number for this time step
        - episode: the actual episode
        """
