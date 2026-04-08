import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from marl.exceptions import CorruptExperimentException
from marl.logging import TIME_STEP_COL, get_logger, Logger, LogSpecs, LogReader
from marl.utils import stats


@dataclass
class Run(LogReader):
    """
    A Run is a single execution of an experiment with a specific seed.

    The `Run` class essentially provides methods to access the metrics and training data of a run.
    """

    rundir: str
    granularity: int

    @staticmethod
    def create_rundir(logdir: str, seed: int):
        now = datetime.now().isoformat().replace(":", "-")
        rundir = os.path.join(logdir, f"run_{now}_seed={seed}")
        os.makedirs(rundir, exist_ok=False)
        return rundir

    def __init__(self, rundir: str, granularity: int, log_specs: LogSpecs):
        super().__init__(rundir)
        self.rundir = rundir
        self.reader = get_logger(rundir, log_specs).reader()
        self.granularity = granularity

    @property
    def seed(self) -> int:
        splits = self.rundir.split("seed=")
        return int(splits[-1])

    @property
    def test_metrics(self):
        return self.reader.test_metrics

    @property
    def train_metrics(self):
        df = self.reader.train_metrics
        if df.is_empty():
            return df
        # Round the time step to match the closest test interval
        df = stats.round_col(df, TIME_STEP_COL, self.granularity)
        # Compute the mean of the metrics for each time step
        df = df.group_by(TIME_STEP_COL).mean()
        return df

    @property
    def training_data(self):
        df = self.reader.training_data
        if df.is_empty():
            return df
        # Make sure we are working with numerical values
        df = stats.ensure_numerical(df, drop_non_numeric=True)
        df = stats.round_col(df, TIME_STEP_COL, self.granularity)
        df = df.group_by(TIME_STEP_COL).agg(pl.col("*").drop_nulls().mean())
        return df

    @property
    def is_running(self) -> bool:
        return self.get_pid() is not None

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

    def get_pid(self):
        pid_file = self.pid_filename
        try:
            with open(pid_file, "r") as f:
                return int(f.read())
        except FileNotFoundError:
            return None


class LiveRun(Logger):
    """
    A *live* run, i.e. a process that is currently executing. This is the preferred way to retrieve the logger of a run while running the experiment, as it ensures that the logger is properly closed and the pid file is removed when the run is finished. It also allows to retrieve the logger reader, which can be useful to monitor the experiment while it is running.
    """

    def __init__(self, logdir: str, seed: int, log_specs: LogSpecs):
        rundir = Run.create_rundir(logdir, seed)
        Logger.__init__(self, rundir)
        pid = os.getpid()
        with open(self.pid_filename, "w") as f:
            f.write(str(pid))
        self.logger = get_logger(rundir, log_specs)

    def __del__(self):
        try:
            os.remove(self.pid_filename)
        except FileNotFoundError:
            pass

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        self.logger.log(data, time_step, prefix)

    def reader(self):
        return self.logger.reader()
