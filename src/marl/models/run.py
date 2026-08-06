import os
from collections.abc import Collection
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from pathlib import Path
from signal import SIGINT, Signals

import numpy as np
import polars as pl
import psutil
from cachetools.func import ttl_cache
from marlenv import MARLEnv

from marl.agents.replay_agent import ReplayAgent
from marl.env import EnvConfig
from marl.logging import TIME_STEP_COL, Logger, LoggerType
from marl.models.trainer import Trainer
from marl.utils import Serializable

RUN_FILE = "run.json"


@dataclass
class LightRun[E: MARLEnv, T: Trainer](Serializable):
    seed: int
    rundir: str
    n_steps: int
    _: KW_ONLY
    test_interval: int = 5_000
    n_tests: int = 1
    loggers: Collection[LoggerType] = ("csv",)
    save_weights: bool = False
    save_actions: bool = True

    @classmethod
    def load(cls, rundir: Path):
        return cls.from_file(rundir / RUN_FILE, exact_type=True)

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
        res = self.pid is not None
        return res

    @ttl_cache(maxsize=1024, ttl=1)
    def latest_train_step(self) -> int:
        try:
            max_train = self.train_metrics.last().select(TIME_STEP_COL).collect().item()
            if max_train >= self.n_steps:
                return max_train
            max_training_data = self.reader.training_data.last().select(TIME_STEP_COL).collect().item()
            return max(max_train, max_training_data)
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.NoDataError):
            return 0

    @ttl_cache(maxsize=1024, ttl=1)
    def latest_test_step(self) -> int:
        try:
            return self.reader.test_metrics.last().select(TIME_STEP_COL).collect().item()
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.NoDataError):
            return 0

    @property
    def is_complete(self):
        return self.latest_time_step >= self.n_steps

    @property
    def latest_time_step(self) -> int:
        if self.is_running:
            return LightRun.latest_train_step(self)
        return LightRun.latest_test_step(self)

    @property
    def progress(self) -> float:
        """The progress between 0 and 1."""
        return self.latest_time_step / self.n_steps

    @property
    def pid(self):
        if not os.path.exists(self.pid_filename):
            return
        try:
            with open(self.pid_filename, "r") as f:
                pid = int(f.read())
            if not psutil.pid_exists(pid):
                self._cleanup_pid_file()
                return
            return pid
        except FileNotFoundError:
            return

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

    def __hash__(self):
        return hash(self.rundir)

    def __eq__(self, other):
        if not isinstance(other, LightRun):
            return False
        return self.rundir == other.rundir

    def __enter__(self):
        return self.to_full().__enter__()

    def to_full(self):
        return Run[E, T].load(self.runpath)


@dataclass
class Run[E: MARLEnv, T: Trainer](LightRun):
    trainer: T
    env: EnvConfig[E]
    test_env: EnvConfig[E]

    @classmethod
    def create(
        cls,
        seed: int,
        rundir: str,
        trainer: T,
        env: EnvConfig[E],
        n_steps: int,
        test_env: EnvConfig[E],
        *,
        test_interval: int = 5_000,
        n_tests: int = 1,
        loggers: Collection[LoggerType] = ("csv",),
        save_weights: bool = True,
        save_actions: bool = True,
    ):
        run = Run(
            seed,
            rundir,
            n_steps,
            trainer,
            env,
            test_env,
            test_interval=test_interval,
            n_tests=n_tests,
            loggers=loggers,
            save_weights=save_weights,
            save_actions=save_actions,
        )
        run.save()
        return run

    def make_agent(self):
        return self.trainer.make_agent()

    def make_replay_agent(self, time_step: int, test_num: int, only_saved_actions: bool):
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
        from marl.runners import compute_test_seed, seeded_rollout

        test_env = self.test_env.make()
        agent = self.make_replay_agent(time_step, test_num, only_saved_actions)
        seed = compute_test_seed(time_step, test_num)
        episode, frames, detailed_actions = seeded_rollout(test_env, agent, seed, compute_frames=True)
        return ReplayEpisode(
            self.runpath, time_step, test_num, episode, frames, detailed_actions, test_env.action_space, agent
        )

    def __hash__(self):
        return hash(self.rundir)

    def __enter__(self):
        if self.is_running:
            raise RuntimeError(f"Run {self.rundir} is already running with pid {self.pid}!")
        pid = os.getpid()
        with open(self.pid_filename, "w") as f:
            f.write(str(pid))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_pid_file()


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
