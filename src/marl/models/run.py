import os
import shutil
import polars as pl
import json
from typing import Optional
from serde.json import to_json
from dataclasses import dataclass
from rlenv.models import Metrics
from marl.utils import CorruptExperimentException
from .replay_episode import ReplayEpisodeSummary
from marl.logging.ws_logger import WSLogger


@dataclass
class Run:
    rundir: str
    seed: int
    pid: Optional[int]
    port: Optional[int]
    current_step: int

    def __init__(self, rundir: str, seed: int, train_df: pl.DataFrame, test_df: pl.DataFrame, train_data: pl.DataFrame):
        """This constructor is not meant to be called directly. Use the static methods `create` or `load` instead."""
        self.rundir = rundir
        self.seed = seed
        self.train_metrics = train_df
        self.test_metrics = test_df
        self.training_data = train_data
        self.port = self.get_port()
        self.pid =  self.get_pid()
        self.current_step = self.get_current_step()

    @staticmethod
    def create(rundir: str, seed: int):
        os.makedirs(rundir, exist_ok=True)
        run = Run(rundir, seed, pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
        with open(os.path.join(rundir, "run.json"), "w") as f:
            f.write(to_json(run))
        return run

    @staticmethod
    def load(rundir: str):
        try:
            train_metrics = pl.read_csv(os.path.join(rundir, "train.csv"))
        except (pl.NoDataError,  FileNotFoundError):
            train_metrics = pl.DataFrame()
        try:
            test_metrics = pl.read_csv(os.path.join(rundir, "test.csv"))
        except (pl.NoDataError, FileNotFoundError):
            test_metrics = pl.DataFrame()
        try:
            train_data = pl.read_csv(os.path.join(rundir, "training_data.csv"))
        except (pl.NoDataError, FileNotFoundError):
            train_data = pl.DataFrame()
        try:
            with open(os.path.join(rundir, "run.json"), "r") as f:
                seed = json.load(f)["seed"]
        except:
            seed = 0
        return Run(rundir, seed, train_metrics, test_metrics, train_data)

    @property
    def is_running(self) -> bool:
        return os.path.exists(os.path.join(self.rundir, "pid"))

    @property
    def latest_checkpoint(self) -> str | None:
        # Get the last test directory
        tests = os.listdir(os.path.join(self.rundir, "test"))
        if len(tests) == 0:
            return None
        last_test = max(tests, key=lambda x: int(x))
        return os.path.join(self.rundir, "test", f"{last_test}")

    def delete(self):
        try:
            shutil.rmtree(self.rundir)
            return
        except FileNotFoundError:
            raise CorruptExperimentException(
                f"Rundir {self.rundir} has already been removed from the file system."
            )

    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        try:
            test_metrics = self.test_metrics.filter(pl.col("time_step") == time_step).sort(
                "timestamp_sec"
            )
            test_dir = os.path.join(self.rundir, "test", f"{time_step}")
            episodes = []
            for i, row in enumerate(test_metrics.rows()):
                episode_dir = os.path.join(test_dir, f"{i}")
                metrics = Metrics(zip(test_metrics.columns, row))
                episode = ReplayEpisodeSummary(episode_dir, metrics)
                episodes.append(episode)
            return episodes
        except pl.ColumnNotFoundError:
            return []

    
    def get_current_step(self) -> int:
        try:
            self.train_metrics = pl.read_csv(os.path.join(self.rundir, "train.csv"))
            max_train = self.train_metrics["time_step"].max()
        except (pl.NoDataError, FileNotFoundError):
            max_train = 0
        try:
            self.test_metrics = pl.read_csv(os.path.join(self.rundir, "test.csv"))
            max_test = self.test_metrics["time_step"].max()
        except (pl.NoDataError, FileNotFoundError):
            max_test = 0
        return max(max_train, max_test)

    def stop(self):
        """Stop the run by sending a SIGINT to the process. This method waits for the process to terminate before returning."""
        pid = self.get_pid()
        if pid is not None:
            import signal

            os.kill(pid, signal.SIGINT)
            print("waiting for process to terminate...")
            os.waitpid(pid, 0)
            print(f"process {pid} has terminated")

    def get_port(self) -> int | None:
        try:
            with open(os.path.join(self.rundir, WSLogger.WS_FILE), "r") as f:
                return int(f.read())
        except FileNotFoundError:
            return None

    def get_pid(self) -> int | None:
        try:
            pid_file = os.path.join(self.rundir, "pid")
            with open(pid_file, "r") as f:
                pid = int(f.read())
                os.kill(pid, 0)
                return pid
        except FileNotFoundError:
            return None
        except ProcessLookupError:
            os.remove(pid_file)
            return None

    