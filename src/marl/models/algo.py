import json
from typing_extensions import Self
from abc import ABC, abstractmethod
import numpy as np
from rlenv.models import Episode, Transition, Observation


class RLAlgo(ABC):
    @abstractmethod
    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        """Get the action to perform given the input observation"""

    def summary(self) -> dict:
        """Dictionary of the relevant algorithm parameters for experiment logging purposes"""
        return {
            "name": self.__class__.__name__
        }

    def save(self, to_path: str):
        """Save the algorithm state to the specified file.
        Saves the algorithm summary by default."""
        with open(to_path, "w") as f:
            json.dump(self.summary(), f)

    def load(self, from_path: str):
        """Load the algorithm state from the specified file."""
        raise NotImplementedError("Not implemented for this algorithm")

    @classmethod
    def from_summary(cls, summary: dict) -> Self:
        """Instantiate the algorithm from its summary"""
        raise NotImplementedError()

    def before_tests(self, time_step: int):
        """Hook before tests, for instance to swap from training to testing policy."""

    def after_tests(self, episodes: list[Episode], time_step: int):
        """
        Hook after tests.
        Subclasses should swap from testing back to training policy.
        """
    def after_step(self, transition: Transition, time_step: int):
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
