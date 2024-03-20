import os
import shutil
import polars as pl
import json
import pickle
from datetime import datetime
from rlenv import Episode, RLEnv
from typing import Optional
from dataclasses import dataclass
from marl.models.algo import RLAlgo
from marl.utils import CorruptExperimentException
from marl import logging
from marl.utils import exceptions
from marl.utils.exceptions import TestEnvNotSavedException
from .replay_episode import ReplayEpisodeSummary


TRAIN = "train.csv"
TEST = "test.csv"
TRAINING_DATA = "training_data.csv"
ENV_PICKLE = "env.pkl"
ACTIONS = "actions.json"
PID = "pid"

# Dataframe columns
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


@dataclass
class Run:
    rundir: str

    def __init__(self, rundir: str):
        """This constructor is not meant to be called directly. Use static methods `create` and `load` instead."""
        self.rundir = rundir

    @staticmethod
    def create(logdir: str, seed: int):
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        rundir = os.path.join(logdir, f"run_{now}_seed={seed}")
        os.makedirs(rundir, exist_ok=False)
        return Run(rundir)

    @staticmethod
    def load(rundir: str):
        return Run(rundir)

    def get_test_env(self, time_step: int, test_num: int) -> RLEnv:
        test_directory = self.test_dir(time_step, test_num)
        env_file = os.path.join(test_directory, ENV_PICKLE)
        try:
            with open(env_file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise TestEnvNotSavedException()

    def get_test_actions(self, time_step: int, test_num: int) -> list[list[int]]:
        test_directory = self.test_dir(time_step, test_num)
        actions_file = os.path.join(test_directory, ACTIONS)
        with open(actions_file, "r") as f:
            return json.load(f)

    def test_dir(self, time_step: int, test_num: Optional[int] = None):
        test_dir = os.path.join(self.rundir, "test", f"{time_step}")
        if test_num is not None:
            test_dir = os.path.join(test_dir, f"{test_num}")
        return test_dir

    def get_saved_algo_dir(self, time_step: int):
        return self.test_dir(time_step)

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
        return self.get_pid() is not None

    @property
    def test_filename(self):
        return os.path.join(self.rundir, TEST)

    @property
    def train_filename(self):
        return os.path.join(self.rundir, TRAIN)

    @property
    def training_data_filename(self):
        return os.path.join(self.rundir, TRAINING_DATA)

    def get_progress(self, max_n_steps: int) -> float:
        try:
            df = self.train_metrics
            return df.select(pl.last(TIME_STEP_COL)).item() / max_n_steps
        except (pl.NoDataError, pl.ColumnNotFoundError):
            return 0.0

    def delete(self):
        try:
            shutil.rmtree(self.rundir)
        except FileNotFoundError:
            raise CorruptExperimentException(f"Rundir {self.rundir} has already been removed from the file system.")

    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        try:
            test_metrics = self.test_metrics.filter(pl.col(TIME_STEP_COL) == time_step).sort(TIMESTAMP_COL)
            test_dir = self.test_dir(time_step)
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
        pid_file = self.pid_filename()
        try:
            with open(pid_file, "r") as f:
                return int(f.read())
        except FileNotFoundError:
            return None

    def pid_filename(self):
        return os.path.join(self.rundir, PID)

    def __enter__(self):
        pid = self.get_pid()
        if pid is not None:
            raise exceptions.AlreadyRunningException(self.rundir, pid)
        with open(self.pid_filename(), "w") as f:
            f.write(str(os.getpid()))
        return RunHandle(
            train_logger=logging.CSVLogger(self.train_filename),
            test_logger=logging.CSVLogger(self.test_filename),
            training_data_logger=logging.CSVLogger(self.training_data_filename),
            run=self,
        )

    def __exit__(self, *args):
        try:
            os.remove(self.pid_filename())
        except FileNotFoundError:
            pass


class RunHandle:
    def __init__(
        self,
        train_logger: logging.CSVLogger,
        test_logger: logging.CSVLogger,
        training_data_logger: logging.CSVLogger,
        run: Run,
    ):
        self.train_logger = train_logger
        self.test_logger = test_logger
        self.training_data_logger = training_data_logger
        self.run = run

    def log_tests(self, episodes: list[Episode], saved_env: list[RLEnv], algo: RLAlgo, time_step: int):
        algo.save(self.run.test_dir(time_step))
        for i, (episode, env) in enumerate(zip(episodes, saved_env)):
            episode_directory = self.run.test_dir(time_step, i)
            self.test_logger.log(episode.metrics, time_step)
            os.makedirs(episode_directory)
            with open(os.path.join(episode_directory, ACTIONS), "w") as a:
                json.dump(episode.actions.tolist(), a)
            if env is not None:
                try:
                    with open(os.path.join(episode_directory, ENV_PICKLE), "wb") as e:
                        pickle.dump(env, e)
                except (AttributeError, pickle.PicklingError):
                    pass

    def log_train_episode(self, episode: Episode, time_step: int, training_logs: dict[str, float]):
        self.train_logger.log(episode.metrics, time_step)
        self.training_data_logger.log(training_logs, time_step)

    def log_train_step(self, metrics: dict[str, float], time_step: int):
        self.training_data_logger.log(metrics, time_step)
