from typing import Any
from dataclasses import dataclass
from marlenv import Episode
from torch import device

from marl.models.trainer import Trainer


@dataclass
class MultiTrainer(Trainer):
    trainers: list[Trainer]

    def __init__(self, *trainers: Trainer):
        if all(t.update_on_episodes for t in trainers):
            update_type = "episode"
        elif all(t.update_on_steps for t in trainers):
            update_type = "step"
        else:
            update_type = "both"

        super().__init__(update_type)
        self.trainers = list(trainers)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int):
        logs = dict[str, Any]()
        for trainer in self.trainers:
            logs.update(trainer.update_episode(episode, episode_num, time_step))
        return logs

    def update_step(self, transition, time_step):
        logs = dict[str, Any]()
        for trainer in self.trainers:
            logs.update(trainer.update_step(transition, time_step))
        return logs

    def to(self, device: device):
        for trainer in self.trainers:
            trainer.to(device)
        return self

    def randomize(self):
        for trainer in self.trainers:
            trainer.randomize()
