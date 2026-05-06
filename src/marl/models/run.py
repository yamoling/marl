import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from signal import SIGINT, Signals
from typing import TYPE_CHECKING, Collection

import numpy as np
import numpy.typing as npt
import polars as pl
import psutil
from cachetools.func import ttl_cache
from marlenv import MARLEnv

from marl.logging import TIME_STEP_COL, Logger, LoggerType
from marl.utils import Serializable, encode_b64_image

if TYPE_CHECKING:
    from marl import Trainer
    from marl.env import EnvConfig

RUN_FILE = "run.json"


@dataclass(unsafe_hash=True)
class Run[E: MARLEnv, T: npt.ArrayLike](Serializable):
    seed: int
    rundir: str
    trainer: Trainer[T]
    env: EnvConfig[E]
    test_env: EnvConfig[E]
    n_steps: int
    test_interval: int
    n_tests: int
    loggers: Collection[LoggerType]
    save_weights: bool = True
    save_actions: bool = True

    def __post_init__(self):
        if not self.runpath.exists():
            self.save()

    @property
    def runpath(self):
        return Path(self.rundir)

    def should_test_at(self, time_step: int):
        if self.n_tests <= 0:
            return False
        # Always test at the last time step, regardless of the test_interval
        if time_step == self.n_steps:
            return True
        if self.test_interval <= 0:
            return False
        return time_step % self.test_interval == 0

    def make_agent(self):
        return self.trainer.make_agent()

    @property
    def run_file(self):
        return self.runpath / RUN_FILE

    @property
    def pid_filename(self):
        return self.runpath / "pid"

    def save(self):
        self.to_file(self.run_file)

    @cached_property
    def logger(self) -> Logger:
        from marl.logging import CSVLogger, MultiLogger, NeptuneLogger, TBLogger, WABLogger

        loggers = list[Logger]()
        for spec in set(self.loggers):
            if spec == "tensorboard":
                loggers.append(TBLogger(self.runpath))
            elif spec == "csv":
                loggers.append(CSVLogger(self.runpath))
            elif spec == "wandb":
                loggers.append(WABLogger(self.runpath))
            elif spec == "neptune":
                loggers.append(NeptuneLogger(self.runpath))
            elif spec == "sqlite":
                raise NotImplementedError("SQLite logger requires additional parameters. Use SQLiteLogger directly.")
            else:
                raise ValueError(f"Unknown logger type: {spec}")
        if len(loggers) == 1:
            return loggers[0]
        return MultiLogger(*loggers)

    @cached_property
    def reader(self):
        return self.logger.reader()

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

    @property
    def latest_train_step(self) -> int:
        try:
            max_train = self.train_metrics.last().select(TIME_STEP_COL).collect().item()
            if max_train == self.n_steps:
                return max_train
            max_training_data = self.reader.training_data.last().select(TIME_STEP_COL).collect().item()
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
    def is_complete(self):
        return self.latest_test_step == self.n_steps

    @property
    def latest_time_step(self) -> int:
        latest_test = self.latest_test_step
        if latest_test == self.n_steps:
            return latest_test
        return max(latest_test, self.latest_train_step)

    @property
    def progress(self) -> float:
        """The progress between 0 and 1."""
        return self.latest_time_step / self.n_steps

    @property
    def pid(self):
        # 1second TTL-cached property
        return _get_pid(self.pid_filename)

    @property
    def ppid(self):
        pid = self.pid
        if pid is None:
            return None
        return psutil.Process(self.pid).ppid()

    def kill(self, signal: Signals | int = SIGINT):
        """Kill the run, if it is running and return whether the run was killed or not."""
        pid = self.pid
        killed = False
        if pid is not None:
            try:
                os.kill(pid, signal)
                killed = True
            except ProcessLookupError:
                pass
        self._cleanup_pid_file()
        return killed

    def _cleanup_pid_file(self):
        try:
            os.remove(self.pid_filename)
        except FileNotFoundError:
            pass

    def __enter__(self):
        if self.is_running:
            raise RuntimeError(f"Run {self.rundir} is already running with pid {self.pid}!")
        pid = os.getpid()
        with open(self.pid_filename, "w") as f:
            f.write(str(pid))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pid_file()

    def make_replay_agent(self, time_step: int, test_num: int, only_saved_actions: bool):
        from marl.models.replay_episode import ReplayAgent

        if only_saved_actions:
            # This should fail if the actions file is not found
            actions = self.get_test_actions(time_step, test_num)
            return ReplayAgent.from_actions_only(actions)
        try:
            # This should **not** fail if the actions file is not found
            actions = self.get_test_actions(time_step, test_num)
            checkpoint_path = self.get_saved_algo_dir(time_step)
            return ReplayAgent.from_agent_and_actions(self.make_agent(), actions, checkpoint_path)
        except FileNotFoundError:
            pass
        try:
            return ReplayAgent.from_agent_only(self.make_agent(), self.get_saved_algo_dir(time_step))
        except FileNotFoundError:
            pass
        try:
            return ReplayAgent.from_actions_only(self.get_test_actions(time_step, test_num))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find any data to replay the episode for time step {time_step} and test number {test_num} in run with seed {self.seed}."
            )

    def replay_episode(self, time_step: int, test_num: int, only_saved_actions: bool):
        from marl.models.replay_episode import ReplayEpisode
        from marl.runners import seeded_rollout

        test_env = self.test_env.make()
        agent = self.make_replay_agent(time_step, test_num, only_saved_actions)
        episode, frames, detailed_actions = seeded_rollout(test_env, agent, self.seed, compute_frames=True)
        frames = [encode_b64_image(f) for f in frames]
        return ReplayEpisode(self.runpath, time_step, test_num, episode, frames, detailed_actions, test_env.action_space, agent)

    @classmethod
    def load(cls, rundir: Path):
        return cls.from_file(rundir / RUN_FILE)


@ttl_cache(ttl=1)
def _get_pid(file: Path):
    try:
        with open(file, "r") as f:
            pid = int(f.read())
        if not psutil.pid_exists(pid):
            _cleanup(file)
            return
        return pid
    except FileNotFoundError:
        return None


def _cleanup(file: Path):
    try:
        os.remove(file)
    except FileNotFoundError:
        pass


def _():
    pass
