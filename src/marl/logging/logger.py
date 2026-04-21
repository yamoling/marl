import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np
import orjson
import polars as pl
from marlenv import Episode, MARLEnv

from marl.models.replay_episode import LightEpisodeSummary

if TYPE_CHECKING:
    from marl import Agent, Trainer

TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


class LogHelper:
    """Helper class for loggers and log readers that provide common methods to get log directories and file paths."""

    def __init__(self, logdir: str):
        self.logdir = logdir

    def get_logdir(self, time_step: int) -> str:
        return os.path.join(self.logdir, str(time_step))

    def test_dir(self, time_step: int):
        test_dir = os.path.join(self.logdir, "test", f"{time_step}")
        return test_dir

    def get_weight_directory(self, time_step: int):
        """Return the file path where the weights of the model are saved for the given time step."""
        return os.path.join(self.logdir, "weights", str(time_step))

    def get_saved_algo_dir(self, time_step: int):
        return self.test_dir(time_step)

    def get_test_actions_file(self, time_step: int):
        test_dir = self.test_dir(time_step)
        return os.path.join(test_dir, "actions.json")


class LogReader(ABC, LogHelper):
    def __init__(self, logdir: str):
        self.logdir = logdir

    @property
    @abstractmethod
    def test_metrics(self) -> pl.LazyFrame: ...

    @property
    @abstractmethod
    def train_metrics(self) -> pl.LazyFrame: ...

    @property
    @abstractmethod
    def training_data(self) -> pl.LazyFrame: ...

    def get_test_episodes(self, time_step: int) -> list[LightEpisodeSummary]:
        try:
            test_metrics = self.test_metrics.filter(pl.col(TIME_STEP_COL) == time_step).drop([TIME_STEP_COL, TIMESTAMP_COL]).collect()
            episodes = []
            for test_num, metrics in enumerate(test_metrics.rows(named=True)):
                episode = LightEpisodeSummary(self.logdir, metrics, time_step, test_num)
                episodes.append(episode)
            return episodes
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.NoDataError):
            # There is no log at all in the file, return an empty list
            return []

    def close(self):
        """Close the log file."""

    def get_test_actions(self, time_step: int) -> list[list[list]]:
        with open(self.get_test_actions_file(time_step), "rb") as f:
            return orjson.loads(f.read())


class Logger(ABC, LogHelper):
    """Logger base class."""

    logdir: str

    def __init__(self, logdir: str):
        super().__init__(logdir)
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        LogHelper.__init__(self, logdir)
        if os.path.exists(logdir):
            if os.path.basename(logdir).lower() in ("test", "tests", "debug"):
                shutil.rmtree(logdir)

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

    def log_params(self, trainer: "Trainer", agent: "Agent", env: MARLEnv, test_env: MARLEnv):
        self.log(asdict(trainer), prefix="params/trainer/", time_step=0)
        self.log(asdict(agent), prefix="params/agent/", time_step=0)
        self.log(asdict(env), prefix="params/env/", time_step=0)
        self.log(asdict(test_env), prefix="params/test_env/", time_step=0)

    @abstractmethod
    def log(self, data: dict[str, Any], time_step: int, prefix: Optional[str] = None): ...

    def log_test_episodes(self, episodes: list[Episode], time_step: int):
        for episode in episodes:
            self.log_test(episode.metrics, time_step)
        actions_file = self.get_test_actions_file(time_step)
        os.makedirs(os.path.dirname(actions_file), exist_ok=True)
        with open(actions_file, "wb") as f:
            f.write(orjson.dumps([e.actions for e in episodes], option=orjson.OPT_SERIALIZE_NUMPY))

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

    def reader(self) -> "LogReader":
        raise NotImplementedError("This method should be implemented by subclasses of Logger to return a LogReader for the same logdir.")

    def save_trainer(self, trainer: "Trainer", time_step: int):
        directory = self.get_saved_algo_dir(time_step)
        if not os.path.exists(directory):
            os.makedirs(directory)
        trainer.save(directory)

    def save_agent(self, agent: "Agent", time_step: int):
        directory = self.get_saved_algo_dir(time_step)
        if not os.path.exists(directory):
            os.makedirs(directory)
        agent.save(directory)
