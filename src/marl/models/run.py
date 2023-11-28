import os
import shutil
import polars as pl
import json
from rlenv import Episode
from typing import Optional
from serde.json import to_json
from dataclasses import dataclass
from marl.utils import CorruptExperimentException
from marl import logging
from marl.utils.stats import agregate_metrics
from .replay_episode import ReplayEpisodeSummary


TRAIN = "train.csv"
TEST = "test.csv"
TRAINING_DATA = "training_data.csv"


@dataclass
class Run:
    rundir: str
    seed: int
    pid: Optional[int]

    def __init__(self, rundir: str, seed: int):
        """This constructor is not meant to be called directly. Use static methods `create` and `load` instead."""
        self.rundir = rundir
        self.seed = seed
        self.train_logger = None
        self.test_logger = None
        self.training_data_logger = None
        self.pid = self.get_pid()

    @staticmethod
    def create(rundir: str, seed: int):
        os.makedirs(rundir, exist_ok=True)
        run = Run(rundir, seed)
        with open(os.path.join(rundir, "run.json"), "w") as f:
            f.write(to_json(run))
        return run

    @staticmethod
    def load(rundir: str):
        with open(os.path.join(rundir, "run.json"), "r") as f:
            seed = json.load(f)["seed"]
        return Run(rundir, seed)

    def log_tests(self, episodes: list[Episode], time_step: int):
        if self.test_logger is None:
            self.test_logger = logging.CSVLogger(self.test_filename)
        agg = agregate_metrics([e.metrics for e in episodes], skip_keys={"timestamp_sec", "time_step"})
        print(agg)
        directory = os.path.join(self.rundir, "test", f"{time_step}")
        for i, episode in enumerate(episodes):
            episode_directory = os.path.join(directory, f"{i}")
            self.test_logger.log(episode.metrics)
            os.makedirs(episode_directory)
            with open(os.path.join(episode_directory, "actions.json"), "w") as a:
                json.dump(episode.actions.tolist(), a)

    def log_train_episode(self, episode: Episode, training_logs: dict):
        if self.train_logger is None:
            self.train_logger = logging.CSVLogger(self.train_filename)
        self.train_logger.log(episode.metrics)
        if len(training_logs) > 1:
            if self.training_data_logger is None:
                self.training_data_logger = logging.CSVLogger(self.training_data_filename)
            self.training_data_logger.log(training_logs)

    def log_train_step(self, metrics: dict):
        if len(metrics) > 1:
            if self.training_data_logger is None:
                self.training_data_logger = logging.CSVLogger(self.training_data_filename)
            self.training_data_logger.log(metrics)

    def test_dir(self, time_step: int):
        return os.path.join(self.rundir, "test", f"{time_step}")

    @property
    def train_metrics(self):
        try:
            # With SMAC, there are sometimes episodes that are not finished and that produce
            # None values for some metrics. We ignore these episodes.
            return pl.read_csv(self.train_filename, ignore_errors=True)
        except (pl.NoDataError, FileNotFoundError):
            return pl.DataFrame()

    @property
    def test_metrics(self):
        try:
            return pl.read_csv(self.test_filename, ignore_errors=True)
        except (pl.NoDataError, FileNotFoundError):
            return pl.DataFrame()

    @property
    def training_data(self):
        try:
            return pl.read_csv(self.training_data_filename)
        except (pl.NoDataError, FileNotFoundError):
            return pl.DataFrame()

    @property
    def is_running(self) -> bool:
        return os.path.exists(os.path.join(self.rundir, "pid"))

    @property
    def test_filename(self):
        return os.path.join(self.rundir, TEST)

    @property
    def train_filename(self):
        return os.path.join(self.rundir, TRAIN)

    @property
    def training_data_filename(self):
        return os.path.join(self.rundir, TRAINING_DATA)

    def delete(self):
        try:
            shutil.rmtree(self.rundir)
            return
        except FileNotFoundError:
            raise CorruptExperimentException(f"Rundir {self.rundir} has already been removed from the file system.")

    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        try:
            test_metrics = self.test_metrics.filter(pl.col("time_step") == time_step).sort("timestamp_sec")
            test_dir = os.path.join(self.rundir, "test", f"{time_step}")
            episodes = []
            for i, row in enumerate(test_metrics.rows()):
                episode_dir = os.path.join(test_dir, f"{i}")
                metrics = dict(zip(test_metrics.columns, row))
                episode = ReplayEpisodeSummary(episode_dir, metrics)
                episodes.append(episode)
            return episodes
        except pl.ColumnNotFoundError:
            # There is no log at all in the file, return an empty list
            return []

    def get_pid(self) -> int | None:
        pid_file = os.path.join(self.rundir, "pid")
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read())
                os.kill(pid, 0)
                return pid
        except FileNotFoundError:
            return None
        except ProcessLookupError:
            os.remove(pid_file)
            return None
