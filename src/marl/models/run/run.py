import os
import math
import shutil
import polars as pl
import orjson
import signal
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from marl.exceptions import (
    CorruptExperimentException,
    RunProcessNotFound,
    NotRunningExcception,
)
from marl.utils import stats
from marl.logging import LogReader, TIME_STEP_COL, TIMESTAMP_COL, CSVLogger
from marl.models.replay_episode import LightEpisodeSummary


TRAIN = "train.csv"
TEST = "test.csv"
TRAINING_DATA = "training_data.csv"
ACTIONS = "actions.json"
PID = "pid"


@dataclass
class Run:
    rundir: str
    seed: int
    n_tests: int
    test_interval: int
    n_steps: int
    """The step up to which the run will be executed, i.e. the max step number."""

    def __init__(self, rundir: str, seed: int, n_tests: int, test_interval: int, n_steps: int, reader: LogReader):
        """This constructor is not meant to be called directly. Use static methods `create` and `load` instead."""
        self.rundir = rundir
        self.seed = seed
        self.n_tests = n_tests
        self.test_interval = test_interval
        self.n_steps = n_steps
        self._reader = reader

    @staticmethod
    def from_experiment(experiment, seed: int, n_tests: int = 1):
        from marl import Experiment

        # For type hinting
        assert isinstance(experiment, Experiment)
        return Run.create(
            logdir=experiment.logdir,
            seed=seed,
            n_tests=n_tests,
            test_interval=experiment.test_interval,
            n_steps=experiment.n_steps,
        )

    @staticmethod
    def create(logdir: str, seed: int, n_tests: int, test_interval: int, n_steps: int):
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
        rundir = os.path.join(logdir, f"run_{now}_seed={seed}")
        os.makedirs(rundir, exist_ok=False)
        reader = CSVLogger.reader(rundir)
        run = Run(rundir, seed, n_tests, test_interval, n_steps, reader)
        with open(os.path.join(rundir, "run.json"), "wb") as f:
            f.write(orjson.dumps(run))
        return run

    @staticmethod
    def load(rundir: str):
        reader = CSVLogger.reader(rundir)
        try:
            with open(os.path.join(rundir, "run.json"), "rb") as f:
                run = orjson.loads(f.read())
            return Run(**run, reader=reader)
        except FileNotFoundError:
            # If there is no run.json file, deduce the parameters from the directory structure
            splits = rundir.split("seed=")
            seed = int(splits[-1])
            test_steps = []
            n_tests = 1
            for folder in os.listdir(os.path.join(rundir, "test")):
                basename = folder
                folder = os.path.join(rundir, "test", folder)
                if os.path.isdir(folder):
                    try:
                        n_tests = max(n_tests, len([t for t in os.listdir(folder) if t.isnumeric()]))
                        test_steps.append(int(basename))
                    except ValueError:
                        pass
            n_steps = max(test_steps)
            test_interval = math.gcd(*test_steps)
            return Run(rundir, seed, n_tests, test_interval, n_steps, reader)

    @staticmethod
    def get_test_seed(time_step: int, test_num: int):
        return time_step + test_num

    def get_test_actions(self, time_step: int, test_num: int) -> list:
        test_directory = self.test_dir(time_step, test_num)
        actions_file = os.path.join(test_directory, ACTIONS)
        with open(actions_file, "r") as f:
            return orjson.loads(f.read())

    def get_train_actions(self, time_step: int) -> list:
        train_directory = self.train_dir(time_step)
        actions_file = os.path.join(train_directory, ACTIONS)
        with open(actions_file, "r") as f:
            return orjson.loads(f.read())

    def test_dir(self, time_step: int, test_num: Optional[int] = None):
        test_dir = os.path.join(self.rundir, "test", f"{time_step}")
        if test_num is not None:
            test_dir = os.path.join(test_dir, f"{test_num}")
        return test_dir

    def train_dir(self, time_step: int):
        return os.path.join(self.rundir, "train", f"{time_step}")

    def get_saved_algo_dir(self, time_step: int):
        return self.test_dir(time_step)

    @property
    def test_metrics(self):
        return self._reader.test_metrics

    @property
    def train_metrics(self):
        df = self._reader.train_metrics
        if df.is_empty():
            return df
        # Round the time step to match the closest test interval
        df = stats.round_col(df, TIME_STEP_COL, self.test_interval)
        # Compute the mean of the metrics for each time step
        df = df.group_by(TIME_STEP_COL).mean()
        return df

    @property
    def training_data(self):
        df = self._reader.training_data
        if df.is_empty():
            return df
        # Make sure we are working with numerical values
        df = stats.ensure_numerical(df, drop_non_numeric=True)
        df = stats.round_col(df, TIME_STEP_COL, self.test_interval)
        df = df.group_by(TIME_STEP_COL).agg(pl.col("*").drop_nulls().mean())
        return df

    @property
    def is_running(self) -> bool:
        return self.get_pid() is not None

    def kill(self):
        """
        Kill the current run.

        If it is not running, raise a NotRunningException.
        If the pid file exists but the process is not found, raise a RunProcessNotFound.
        """
        pid = self.get_pid()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGINT)
                return
            except ProcessLookupError:
                raise RunProcessNotFound(self.rundir, pid)
        raise NotRunningExcception(self.rundir)

    @property
    def latest_train_step(self) -> int:
        try:
            return self.train_metrics.select(pl.last(TIME_STEP_COL)).item()
        except pl.exceptions.ColumnNotFoundError:
            return 0

    @property
    def latest_test_step(self) -> int:
        try:
            return self.test_metrics.select(pl.last(TIME_STEP_COL)).item()
        except pl.exceptions.ColumnNotFoundError:
            return 0

    @property
    def latest_time_step(self) -> int:
        return max(self.latest_test_step, self.latest_train_step)

    def get_progress(self, max_n_steps: int) -> float:
        return self.latest_time_step / max_n_steps

    def delete(self):
        try:
            shutil.rmtree(self.rundir)
        except FileNotFoundError:
            raise CorruptExperimentException(f"Rundir {self.rundir} has already been removed from the file system.")

    def get_test_episodes(self, time_step: int) -> list[LightEpisodeSummary]:
        try:
            test_metrics = self.test_metrics.filter(pl.col(TIME_STEP_COL) == time_step).sort(TIMESTAMP_COL)
            test_metrics = test_metrics.drop([TIME_STEP_COL, TIMESTAMP_COL])
            episodes = []
            for test_num, row in enumerate(test_metrics.rows()):
                episode_dir = self.test_dir(time_step, test_num)
                metrics = dict(zip(test_metrics.columns, row))
                episode = LightEpisodeSummary(episode_dir, metrics)
                episodes.append(episode)
            return episodes
        except pl.exceptions.ColumnNotFoundError:
            # There is no log at all in the file, return an empty list
            return []

    def get_pid(self) -> int | None:
        pid_file = self.pid_filename
        try:
            with open(pid_file, "r") as f:
                return int(f.read())
        except FileNotFoundError:
            return None

    @property
    def pid_filename(self):
        return os.path.join(self.rundir, PID)

    @property
    def test_dirs(self):
        """
        Ordered test directories from t=0 to t=n_steps.
        """
        for t in range(0, self.n_steps, self.test_interval):
            yield self.test_dir(t)
