from abc import ABC, abstractmethod

from marl.models import Trainer
from rlenv import Transition, Episode
from marl.qlearning import CNetAlgo, EpisodeCommWrapper

from dataclasses import dataclass
from serde import serialize
from typing import Literal, Any
from typing_extensions import Self

import numpy as np

import torch


@serialize
@dataclass
class CNetTrainer(Trainer):
    def __init__(self, opt, agents: CNetAlgo):
        super().__init__("episode", opt.bs)
        self.opt = opt
        self.agents = agents
        self.memory = EpisodeCommWrapper()

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        if transition.done:
            # Remove the stored hidden and messages
            self.agents.clear_last_record()
        return {}

    def fill_episode(self, episode: Episode, agent_episode):
        agent_episode.steps = episode.episode_len

        for time_step in range(episode.episode_len):
            # Add rewards
            reward = episode.rewards[time_step]
            agent_episode.step_records[time_step].r_t = torch.tensor([reward for _ in range(self.opt.game_nagents)])
            # Add terminals
            done = episode.dones[time_step]
            agent_episode.step_records[time_step].terminal = torch.tensor(done)

        return agent_episode

    def _update(self, time_step: int) -> dict[str, Any]:
        self.agents.learn_from_episode(self.memory.get_batch(self.opt, self.device))
        self.memory.clear()
        logs = self.agents.policy.update(time_step)
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        episode_from_agent = self.agents.get_episode()
        self.memory.add_episode(self.fill_episode(episode, episode_from_agent))

        if (episode_num + 1) % self.update_interval == 0:
            self._update(time_step)

        return {}

    def to(self, device: torch.device):
        """Send the tensors to the given device."""
        self.agents.to(device)
        self.device = device

    def randomize(self):
        """Randomize the state of the trainer."""
        pass
