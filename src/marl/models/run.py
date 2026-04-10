import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from signal import SIGINT, Signals
from functools import cached_property
import psutil

import polars as pl

from marl.exceptions import CorruptExperimentException
from marl.logging import TIME_STEP_COL, LogSpecs, get_logger
from marl.utils import stats

PID_FILENAME = "pid"


@dataclass
class Run:
    """
    A Run is a single execution of an experiment with a specific seed.

    The `Run` class essentially provides methods to access the metrics and training data of a run.
    """

    rundir: str
    log_specs: LogSpecs

    @staticmethod
    def load(rundir: str, log_specs: LogSpecs):
        return Run(rundir, log_specs)

    @staticmethod
    def create(logdir: str, seed: int, log_specs: LogSpecs):
        now = datetime.now().isoformat().replace(":", "-")
        rundir = os.path.join(logdir, f"run_{now}_seed={seed}")
        os.makedirs(rundir, exist_ok=False)
        return Run(rundir, log_specs)

    @cached_property
    def reader(self):
        return get_logger(self.rundir, self.log_specs).reader()

    def test_dir(self, time_step: int, test_num: int | None = None):
        return self.reader.test_dir(time_step, test_num)

    def get_saved_algo_dir(self, time_step: int):
        return self.reader.get_saved_algo_dir(time_step)

    def get_test_episodes(self, time_step: int):
        return self.reader.get_test_episodes(time_step)

    @property
    def seed(self) -> int:
        splits = self.rundir.split("seed=")
        return int(splits[-1])

    @property
    def test_metrics(self):
        return self.reader.test_metrics

    def train_metrics(self, granularity: int):
        """
        Return the training metrics aggregated by time step, where the time steps are rounded to the closest multiple of the given granularity.

        E.g.: if the time steps are [1, 2, 3, 4, 5] and the granularity is 2, the time steps will be rounded to [0, 2, 2, 4, 4], and the metrics will be averaged for each time step, resulting in a dataframe with time steps [0, 2, 4].
        """
        df = self.reader.train_metrics
        if df.is_empty():
            return df
        # Round the time step to match the closest test interval
        df = stats.round_col(df, TIME_STEP_COL, granularity)
        # Compute the mean of the metrics for each time step
        df = df.group_by(TIME_STEP_COL).mean()
        return df

    def training_data(self, granularity: int):
        """
        Return the training data aggregated by time step, where the time steps are rounded to the closest multiple of the given granularity.

        E.g.: if the time steps are [1, 2, 3, 4, 5] and the granularity is 2, the time steps will be rounded to [0, 2, 2, 4, 4], and the metrics will be averaged for each time step, resulting in a dataframe with time steps [0, 2, 4].
        """
        df = self.reader.training_data
        if df.is_empty():
            return df
        # Make sure we are working with numerical values
        df = stats.ensure_numerical(df, drop_non_numeric=True)
        df = stats.round_col(df, TIME_STEP_COL, granularity)
        df = df.group_by(TIME_STEP_COL).agg(pl.col("*").drop_nulls().mean())
        return df

    @property
    def is_running(self) -> bool:
        return self.pid is not None

    def is_completed(self, n_steps: int) -> bool:
        return self.get_progress(n_steps) >= 1.0

    @property
    def latest_train_step(self) -> int:
        try:
            max_train = self.reader.train_metrics[TIME_STEP_COL].max()
            if max_train is None:
                max_train = 0
            assert isinstance(max_train, int)
            max_training_data = self.reader.training_data[TIME_STEP_COL].max()
            if max_training_data is None:
                max_training_data = 0
            assert isinstance(max_training_data, int)
            return max(max_train, max_training_data)
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

    @property
    def pid_filename(self):
        return os.path.join(self.rundir, PID_FILENAME)

    def _cleanup_pid_file(self):
        try:
            os.remove(self.pid_filename)
        except FileNotFoundError:
            pass

    @property
    def pid(self):
        pid_file = self.pid_filename
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read())
            if not psutil.pid_exists(pid):
                self._cleanup_pid_file()
                return
            return pid
        except FileNotFoundError:
            return None

    def get_parent_pid(self):
        pid = self.pid
        if pid is None:
            return None
        return psutil.Process(self.pid).ppid()

    def kill(self, signal: Signals | int = SIGINT):
        if not isinstance(signal, int):
            signal = int(signal)
        pid = self.pid
        if pid is not None:
            try:
                os.kill(pid, signal)
            except ProcessLookupError:
                pass
        self._cleanup_pid_file()

    def __enter__(self):
        if self.is_running:
            raise RuntimeError(f"Run {self.rundir} is already running with pid {self.pid}!")
        pid = os.getpid()
        with open(self.pid_filename, "w") as f:
            f.write(str(pid))
        return get_logger(self.rundir, self.log_specs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pid_file()
