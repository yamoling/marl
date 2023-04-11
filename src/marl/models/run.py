import os
import json
import shutil
import polars as pl
from dataclasses import dataclass
from rlenv.models import Metrics
from marl.utils import CorruptExperimentException
from .replay_episode import ReplayEpisodeSummary
from marl.logging.ws_logger import WSLogger


@dataclass
class Run:
    rundir: str
    train_metrics: pl.DataFrame
    test_metrics: pl.DataFrame

    def __init__(self, rundir: str, train_df: pl.DataFrame, test_df: pl.DataFrame):
        """This constructor is not meant to be called directly. Use the static methods `create` or `load` instead."""
        self.rundir = rundir
        self.train_metrics = train_df
        self.test_metrics = test_df

    @staticmethod
    def create(rundir: str):
        os.makedirs(rundir, exist_ok=True)
        return Run(rundir, pl.DataFrame(), pl.DataFrame())

    @staticmethod
    def load(rundir):
        train_metrics = pl.read_csv(os.path.join(rundir, "train.csv"))
        test_metrics = pl.read_csv(os.path.join(rundir, "test.csv"))
        return Run(rundir, train_metrics, test_metrics)

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

    @property
    def current_step(self) -> int:
        try:
            with open(os.path.join(self.rundir, "run.json"), "r") as f:
                return json.load(f)["current_step"]
        except FileNotFoundError:
            return 0

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

    def to_json(self) -> dict[str, str | int | None]:
        return {"rundir": self.rundir, "port": self.get_port(), "pid": self.get_pid()}
