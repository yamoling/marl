from abc import ABC, abstractmethod

from marl.models import Trainer
from rlenv import Transition, Episode
from marl.qlearning import CNetAlgo, EpisodeCommWrapper

from dataclasses import dataclass
from serde import serialize
from typing import Literal, Any
from typing_extensions import Self

import torch

@serialize
@dataclass
class CNetTrainer(Trainer):
    def __init__(self, opt,  agents: CNetAlgo):
        super().__init__("episode", opt.bs)
        self.opt = opt
        self.agents = agents
        self.memory = EpisodeCommWrapper()

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        return {}
    
    def fill_episode(self, episode: Episode, agent_episode):
        agent_episode.steps = episode.episode_len

        for time_step in range(episode.episode_len):
            # Add rewards
            reward = episode.rewards[time_step]
            agent_episode.step_records[time_step].rewards = reward # TODO convert to the good type
            # Add terminals
            done = episode.dones[time_step]
            agent_episode.step_records[time_step].terminal = done
        
        return agent_episode

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:

        episode_from_agent = self.agents.get_episode()
        self.memory.add_episode(self.fill_episode(episode, episode_from_agent))

        if episode_num % self.update_interval == 0:
            self.agents.learn_from_episode(self.memory.get_batch(self.opt))
            self.memory.clear()

        return {}


    def to(self, device: torch.device) -> Self:
        """Send the tensors to the given device."""
        self.agents.to(device)

    def randomize(self):
        """Randomize the state of the trainer."""
        pass


