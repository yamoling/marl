from dataclasses import dataclass
from typing import Any

import torch
from marlenv import Episode, Observation, State, Transition

from marl import Trainer


@dataclass
class MultiTrainer(Trainer):
    def __init__(self, *trainers: Trainer):
        super().__init__()
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

    def randomize(self, method="xavier"):
        for trainer in self.trainers:
            trainer.randomize(method)

    def value(self, obs: Observation, state: State):
        return [t.value(obs, state) for t in self.trainers]

    def save(self, directory_path: str):
        for i, trainer in enumerate(self.trainers):
            trainer.save(f"{directory_path}/{trainer.name}-{i}")
