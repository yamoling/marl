from typing import Any

import numpy as np
import polars as pl
from marlenv import MARLEnv
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from marl.models.agent import Agent
from marl.models.trainer import Trainer

from .logger import Logger, LogReader


class TBReader(LogReader):
    def __init__(self, weight_path: str, log_file: str):
        super().__init__(weight_path)
        self.log_file = log_file

    def parse_tensorboard(self, prefix: str):
        """returns a dictionary of pandas dataframes for each requested scalar"""
        ea = event_accumulator.EventAccumulator(self.log_file)
        ea.Reload()
        keys = [k for k in ea.Tags()["scalars"] if k.startswith(prefix)]
        cols = {k: ea.Scalars(k) for k in keys}
        return pl.LazyFrame(cols)

    @property
    def test_metrics(self):
        """Return all the metrics whose prefix is 'test/'."""
        return self.parse_tensorboard("test/")

    @property
    def train_metrics(self):
        """Return all the metrics whose prefix is 'train/'."""
        return self.parse_tensorboard("train/")

    @property
    def training_data(self):
        """Return all the metrics whose prefix is 'training-data/'."""
        return self.parse_tensorboard("training-data/")


class TBLogger(Logger):
    def __init__(self, logdir: str):
        super().__init__(logdir)
        self.writer = SummaryWriter(logdir)

    def log_params(self, trainer: Trainer, agent: Agent, env: MARLEnv, test_env: MARLEnv):
        return

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        if prefix is None:
            prefix = ""
        for key, value in data.items():
            match value:
                case float() | int() | bool() | np.floating() | np.int64():
                    self.writer.add_scalar(f"{prefix}{key}", value, time_step)
                case dict():
                    self.log(value, time_step, f"{prefix}{key}")
                case str():
                    self.writer.add_text(f"{prefix}{key}", value, time_step)
                case _:
                    raise NotImplementedError(f"Unsupported data type for key {key}: {type(value)} with value {value}")

    def reader(self):
        return TBReader(self.logdir, self.logdir)
