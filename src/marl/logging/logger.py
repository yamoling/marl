import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Optional

import cv2
import numpy as np
import orjson
import polars as pl
from marlenv import Episode

from marlenv import MARLEnv
from marl.agents import Agent
from marl.models.trainer import Trainer

ACTIONS = "actions.json"
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


class LogReader(ABC):
    def __init__(self, weight_path: str):
        super().__init__()
        self.weight_path = weight_path

    @property
    @abstractmethod
    def test_metrics(self) -> pl.DataFrame: ...

    @property
    @abstractmethod
    def train_metrics(self) -> pl.DataFrame: ...

    @property
    @abstractmethod
    def training_data(self) -> pl.DataFrame: ...

    def get_weights_directory(self, time_step: int) -> str:
        """Return the file path where the weights of the model are saved."""
        return os.path.join(self.weight_path, str(time_step))

    def close(self):
        """Close the log file."""


class Logger(ABC):
    """Logger base class."""

    logdir: str

    def __init__(self, logdir: str):
        super().__init__()
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        self.logdir = logdir
        if os.path.exists(logdir):
            if os.path.basename(logdir).lower() in ("test", "tests", "debug"):
                shutil.rmtree(logdir)

    def get_logdir(self, time_step: int) -> str:
        return os.path.join(self.logdir, str(time_step))

    def log_train(self, data: dict[str, Any], time_step: int):
        if len(data) == 0:
            return
        self.log(data, time_step, prefix="train/")

    def log_training_data(self, data: dict[str, Any], time_step: int):
        if len(data) == 0:
            return
        self.log(data, time_step, prefix="training-data/")

    def log_test(self, data: dict[str, Any], time_step: int):
        if len(data) == 0:
            return
        self.log(data, time_step, prefix="test/")

    def log_params(self, trainer: Trainer, agent: Agent, env: MARLEnv, test_env: MARLEnv):
        self.log(asdict(trainer), prefix="params/trainer/", time_step=0)
        self.log(asdict(agent), prefix="params/agent/", time_step=0)
        self.log(asdict(env), prefix="params/env/", time_step=0)
        self.log(asdict(test_env), prefix="params/test_env/", time_step=0)

    @abstractmethod
    def log(self, data: dict[str, Any], time_step: int, prefix: Optional[str] = None): ...

    def test_dir(self, time_step: int, test_num: Optional[int] = None):
        test_dir = os.path.join(self.logdir, "test", f"{time_step}")
        if test_num is not None:
            test_dir = os.path.join(test_dir, f"{test_num}")
        return test_dir

    def log_test_episodes(self, episodes: list[Episode], time_step: int):
        for i, episode in enumerate(episodes):
            episode_directory = self.test_dir(time_step, i)
            self.log_test(episode.metrics, time_step)
            if os.path.exists(episode_directory):
                print(f"Warning: episode directory {episode_directory} already exists ! Overwriting...")
            else:
                os.makedirs(episode_directory)
            with open(os.path.join(episode_directory, ACTIONS), "wb") as f:
                bytes_data = orjson.dumps(episode.actions, option=orjson.OPT_SERIALIZE_NUMPY)
                f.write(bytes_data)

    def log_as_json(self, object: object, time_step: int, name: Optional[str] = None):
        directory = self.get_logdir(time_step)
        if name is None:
            name = object.__class__.__name__
        filename = os.path.join(directory, f"{name}.json")
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(directory, f"{name}-{counter}.json")
            counter += 1
        with open(filename, "wb") as f:
            f.write(orjson.dumps(object, option=orjson.OPT_SERIALIZE_NUMPY))

    def log_image(self, image: np.ndarray, time_step: int, filename: str):
        directory = self.get_logdir(time_step)
        if not filename.endswith(".png") and not filename.endswith(".jpg"):
            filename += ".png"
        filename = os.path.join(directory, filename)
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(directory, f"image-{counter}.png")
            counter += 1
        cv2.imwrite(filename, image)

    @staticmethod
    @abstractmethod
    def reader(from_directory: str) -> "LogReader": ...
