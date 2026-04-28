import random
from dataclasses import dataclass

import numpy as np
from marlenv import Observation

from marl.models import Agent, AgentWrapper, ContextualBandit


@dataclass
class MAVENAgent(AgentWrapper[np.int64]):
    noise_size: int

    def __init__(self, noise_size: int, workers: Agent[np.int64], bandit: ContextualBandit[np.int64]):
        super().__init__(workers)
        self._episode_noise = None
        self._saved_episode_noise = None
        self.noise_size = noise_size
        self.bandit = bandit

    def choose_action(self, observation: Observation, *, with_details: bool = False):
        if self._episode_noise is None:
            self._episode_noise = self.bandit.choose_action(observation=observation).astype(np.float32)
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

    def seed(self, seed: int):
        random.seed(seed)
        super().seed(seed)
