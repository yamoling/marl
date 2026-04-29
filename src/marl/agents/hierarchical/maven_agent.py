import numpy as np
import numpy.typing as npt
from marlenv import Observation

from marl.models import Agent, HierarchicalAgent

from ..simple_agent import DiscreteOneHotAgent


class MAVENAgent(HierarchicalAgent[npt.NDArray[np.int64], npt.NDArray[np.int64]]):
    noise_size: int

    def __init__(self, noise_size: int, workers: Agent[npt.NDArray[np.int64]], meta_agent: DiscreteOneHotAgent):
        super().__init__(meta_agent, workers)
        self._episode_noise = None
        self._saved_episode_noise = None
        self.noise_size = noise_size
        self.meta_agent = meta_agent

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        if self._episode_noise is None:
            noise = self.meta_agent.choose_action(observation=observation).action
            self._episode_noise = noise.astype(np.float32)
        observation.extras[:, -self.noise_size :] = self._episode_noise
        action = super().choose_action(observation, with_details=with_details)
        action["maven-noise"] = self._episode_noise
        return action

    def new_episode(self):
        self._episode_noise = None
        return super().new_episode()

    def set_testing(self):
        # Save the episode noise from the training
        self._saved_episode_noise = self._episode_noise
        return super().set_testing()

    def set_training(self):
        # Restore the episode noise from the training
        self._episode_noise = self._saved_episode_noise
        return super().set_training()
