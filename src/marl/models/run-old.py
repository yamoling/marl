import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from signal import SIGINT, Signals

import numpy as np
import polars as pl
import psutil

from marl.exceptions import CorruptExperimentException
from marl.logging import TIME_STEP_COL, TIMESTAMP_COL, LogSpecs, get_logger

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

    def test_dir(self, time_step: int):
        return self.reader.test_dir(time_step)

    def get_saved_algo_dir(self, time_step: int):
        return self.reader.get_saved_algo_dir(time_step)

    def get_test_episodes(self, time_step: int):
        return self.reader.get_test_episodes(time_step)

    def get_test_actions(self, time_step: int, test_num: int):
        all_actions = self.reader.get_test_actions(time_step)
        return np.array(all_actions[test_num])

    @property
    def test_metrics(self):
        return self.reader.test_metrics

    @property
    def train_metrics(self):
        return self.reader.train_metrics

    @property
    def training_data(self):
        return self.reader.training_data

    @property
    def is_running(self) -> bool:
        return self.pid is not None

    def is_completed(self, n_steps: int) -> bool:
        return self.get_progress(n_steps) >= 1.0

    @property
    def latest_train_step(self) -> int:
        try:
            max_train = self.reader.train_metrics.last().select(TIME_STEP_COL).collect().item()
            if max_train is None:
                max_train = 0
            assert isinstance(max_train, int)
            max_training_data = self.reader.training_data.last().select(TIME_STEP_COL).collect().item()
            if max_training_data is None:
                max_training_data = 0
            assert isinstance(max_training_data, int)
            return max(max_train, max_training_data)
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.NoDataError):
            return 0

    @property
    def latest_test_step(self) -> int:
        try:
            return self.reader.test_metrics.last().select(TIME_STEP_COL).collect().item()
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.NoDataError):
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
