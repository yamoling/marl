import os
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, Sequence

import torch
from marlenv import Episode, Observation, State, Transition

from marl.utils import Serializable

from .agent import Agent
from .nn import NN, IRModule, randomize


@dataclass
class Trainer[T](Serializable):
    """Algorithm trainer class."""

    _: KW_ONLY
    gamma: float = 0.99
    ir_module: IRModule | None = None
    grad_norm_clipping: float | None = None
    batch_size: int = 64
    train_interval: tuple[int, Literal["step", "episode"]] = (5, "step")

    def __post_init__(self):
        super().__post_init__()
        self._device = torch.device("cpu")

    def make_agent(self) -> Agent[T]:
        raise NotImplementedError("Trainer must implement make_agent method")

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        """
        Update to call after each step. Should be run when update_after_each == "step".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        """
        Update to call after each episode. Should be run when update_after_each == "episode".

        Returns:
            dict[str, Any]: A dictionary of training metrics to log.
        """
        return {}

    def value(self, obs: Observation, state: State) -> float | Sequence:
        """
        Compute the value of the current state or observation.
        """
        return 0.0

    def save(self, directory: Path):
        if not directory.exists():
            os.makedirs(directory)
        for nn in self.networks():
            nn.save(directory)

    def load(self, directory: Path):
        for nn in self.networks():
            nn.load(directory)

    @property
    def device(self):
        return self._device

    def networks(self):
        """Dynamic list of neural networks attributes in the trainer"""
        return [nn for nn in self.__dict__.values() if isinstance(nn, NN)]

    def randomize(self, method: Literal["xavier", "orthogonal"] = "xavier"):
        """Randomize the parameters of all the neural networks in the trainer."""

        for nn in self.networks():
            if isinstance(nn, NN):
                nn.randomize(method)
            else:
                randomize(torch.nn.init.xavier_uniform_, nn)

    def to(self, device: torch.device) -> Self:
        """Send the networks to the given device."""
        self._device = device
        for nn in self.networks():
            nn.to(device)
        return self


@dataclass
class HierarchicalTrainer[T, T1: Trainer, T2: Trainer](Trainer[T]):
    meta_trainer: T1 = field(init=False)
    worker_trainer: T2 = field(init=False)

    def update_step(self, transition: Transition, time_step: int) -> dict[str, Any]:
        meta_logs = self.meta_trainer.update_step(transition, time_step)
        worker_logs = self.worker_trainer.update_step(transition, time_step)
        return self.merge_logs(worker_logs, meta_logs)

    def update_episode(self, episode: Episode, episode_num: int, time_step: int) -> dict[str, Any]:
        meta_logs = self.meta_trainer.update_episode(episode, episode_num, time_step)
        worker_logs = self.worker_trainer.update_episode(episode, episode_num, time_step)
        return self.merge_logs(worker_logs, meta_logs)

    def networks(self):
        return self.meta_trainer.networks() + self.worker_trainer.networks()

    def to(self, device: torch.device) -> Self:
        self.meta_trainer.to(device)
        self.worker_trainer.to(device)
        return self

    def value(self, obs: Observation, state: State):
        return self.meta_trainer.value(obs, state)

    @staticmethod
    def merge_logs(worker_logs: dict, meta_logs: dict):
        return {
            **{f"meta/{key}": value for key, value in meta_logs.items()},
            **{f"worker/{key}": value for key, value in worker_logs.items()},
        }
