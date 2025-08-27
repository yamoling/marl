from abc import ABC, abstractmethod
from typing import Literal, Optional
from dataclasses import dataclass
from pprint import pprint
from marlenv import Episode
import numpy as np
import os
import cv2
import orjson
import polars as pl
import shutil

ACTIONS = "actions.json"
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


@dataclass
class LogWriter(ABC):
    filename: str
    quiet: bool

    @abstractmethod
    def log(self, data: dict[str, float], time_step: int):
        """Log the data."""

    def log_print(self, data: dict[str, float], time_step: int):
        """Log and print the data."""
        self.log(data, time_step)
        if not self.quiet:
            pprint(data)

    @abstractmethod
    def close(self):
        """Close the log file."""


class LogReader(ABC):
    @property
    @abstractmethod
    def test_metrics(self) -> pl.DataFrame: ...

    @property
    @abstractmethod
    def train_metrics(self) -> pl.DataFrame: ...

    @property
    @abstractmethod
    def training_data(self) -> pl.DataFrame: ...

    def close(self):
        """Close the log file."""


@dataclass
class Logger(ABC):
    """Logging interface"""

    logdir: str
    quiet: bool
    test: LogWriter
    train: LogWriter
    training_data: LogWriter

    def __init__(self, logdir: str, quiet: bool, test: LogWriter, train: LogWriter, training_data: LogWriter):
        super().__init__()
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        self.logdir = logdir
        if os.path.exists(logdir):
            if os.path.basename(logdir).lower() in ["test", "debug"]:
                shutil.rmtree(logdir)
        os.makedirs(logdir, exist_ok=True)
        self.quiet = quiet
        self.test = test
        self.train = train
        self.training_data = training_data

    def get_logdir(self, time_step: int) -> str:
        return os.path.join(self.logdir, str(time_step))

    def log(self, kind: Literal["test", "train", "training_data"], data: dict[str, float], time_step: int):
        """Log the data."""
        match kind:
            case "test":
                self.test.log(data, time_step)
            case "train":
                self.train.log(data, time_step)
            case "training_data":
                self.training_data.log(data, time_step)

    def test_dir(self, time_step: int, test_num: Optional[int] = None):
        test_dir = os.path.join(self.logdir, "test", f"{time_step}")
        if test_num is not None:
            test_dir = os.path.join(test_dir, f"{test_num}")
        return test_dir

    def log_tests(self, episodes: list[Episode], time_step: int):
        for i, episode in enumerate(episodes):
            episode_directory = self.test_dir(time_step, i)
            self.test.log(episode.metrics, time_step)
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

    def __del__(self):
        self.test.close()
        self.train.close()
        self.training_data.close()
