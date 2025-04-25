from typing import Any
from marlenv import Episode, Observation, State, Transition
import torch
from marl import Trainer
from dataclasses import dataclass


@dataclass
class MultiTrainer(Trainer):
    def __init__(self, /, *trainers: Trainer, device: torch.device | None = None):
        super().__init__(device)
        self.trainers = trainers

    def update_step(self, transition: Transition, time_step: int):
        logs = dict[str, Any]()
        for trainer in self.trainers:
            trainer_logs = trainer.update_step(transition, time_step)
            for key, value in trainer_logs.items():
                logs[f"{trainer.name}/{key}"] = value
        return logs

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        logs = dict[str, Any]()
        for trainer in self.trainers:
            trainer_logs = trainer.update_episode(episode, episode_num, time_step)
            for key, value in trainer_logs.items():
                logs[f"{trainer.name}/{key}"] = value
        return logs

    def to(self, device: torch.device):
        for trainer in self.trainers:
            trainer.to(device)
        return self

    def randomize(self):
        for trainer in self.trainers:
            trainer.randomize()
        return self

    def value(self, obs: Observation, state: State):
        return [t.value(obs, state) for t in self.trainers]
