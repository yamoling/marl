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
        """Save the algorithm state to the specified file."""
        raise NotImplementedError()

    def load(self, from_path: str):
        """Load the algorithm state from the specified file."""
        raise NotImplementedError()

    def before_tests(self):
        """Hook before tests, for instance to swap from training to testing policy."""

    def after_tests(self, episodes: list[Episode], time_step: int):
        """
        Hook after tests.
        Subclasses should swap from testing back to training policy.
        """
    def after_step(self, transition: Transition, step_num: int):
        """Hook after every training step."""

    def before_episode(self, episode_num: int):
        """Hook before every training and testing episode."""

    def after_episode(self, episode_num: int, episode: Episode):
        """Hook after every training episode."""


class RLAlgoWrapper(RLAlgo):
    def __init__(self, wrapped: RLAlgo) -> None:
        super().__init__()
        self.algo = wrapped

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        return self.algo.choose_action(observation)

    def save(self, to_path: str):
        return self.algo.save(to_path)

    def load(self, from_path: str):
        return self.algo.load(from_path)

    def before_episode(self, episode_num: int):
        return self.algo.before_episode(episode_num)
    
    def before_tests(self):
        return self.algo.before_tests()

    def after_episode(self, episode_num: int, episode: Episode):
        return self.algo.after_episode(episode_num, episode)

    def after_step(self, transition: Transition, step_num: int):
        return self.algo.after_step(transition, step_num)

    def after_tests(self, episodes: list[Episode], time_step: int):
        return self.algo.after_tests(episodes, time_step)