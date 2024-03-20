from abc import ABC, abstractmethod

from marl.models import Trainer
from rlenv import Transition, Episode
from marl.qlearning import CNetAlgo

from dataclasses import dataclass
from serde import serialize
from typing import Literal, Any
from typing_extensions import Self

import torch


class EpisodeCommWrapper:
    def __init__(self):
        self.episodes = []

    def get_episode(self, ep_num: int):
        if len(self.episodes) > ep_num:
            return self.episodes[ep_num]
    
    def add_episode(self, episode: Episode):
        self.episodes.append(episode)
    
    def clear(self):
        self.episodes = []

@serialize
@dataclass
class CNetTrainer(Trainer):
    def __init__(self, opt,  agents: CNetAlgo):
        super().__init__("episode", opt.bs)
        self.agents = agents
        self.memory = EpisodeCommWrapper()
        self.current_episode = None

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        
        Put information in memory (transition + agent)
        """
        if (self.current_episode is None):
            self.current_episode = self.agents.create_episode()
        # TODO : update current episode
        
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.

        When update interval is reached : update with the bs last episodes
        """
        # TODO : Add current_episode to memory
        # clear current_episode
        # if update_interval is reached, 
            # update with the memory
            # clear memory
        
        return {}


    def to(self, device: torch.device) -> Self:
        """Send the tensors to the given device."""
        self.agents.to(device)

    def randomize(self):
        """Randomize the state of the trainer."""
        pass


