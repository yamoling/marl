import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from signal import SIGINT, Signals
from typing import TYPE_CHECKING, Collection

import numpy as np
import numpy.typing as npt
import psutil
from marlenv import Space

from marl.logging import Logger, LoggerType
from marl.utils import Serializable, encode_b64_image

if TYPE_CHECKING:
    from marl.config import EnvConfig, TrainerConfig


@dataclass
class Run[A: Space, T: npt.ArrayLike](Serializable):
    seed: int
    rundir: Path
    trainer: TrainerConfig[T]
    env: EnvConfig[A]
    test_env: EnvConfig[A]
    n_steps: int
    test_interval: int
    n_tests: int
    loggers: Collection[LoggerType]
    save_weights: bool = True
    save_actions: bool = True

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
        return self.trainer.make().make_agent()

    @property
    def run_file(self):
        return self.rundir / "run.json"

    @property
    def pid_filename(self):
        return self.rundir / "pid"

    def save(self):
        self.to_file(self.run_file)

    @cached_property
    def logger(self) -> Logger:
        from marl.logging import CSVLogger, MultiLogger, NeptuneLogger, TBLogger, WABLogger

        loggers = list[Logger]()
        for spec in set(self.loggers):
            if spec == "tensorboard":
                loggers.append(TBLogger(self.rundir))
            elif spec == "csv":
                loggers.append(CSVLogger(self.rundir))
            elif spec == "wandb":
                loggers.append(WABLogger(self.rundir))
            elif spec == "neptune":
                loggers.append(NeptuneLogger(self.rundir))
            elif spec == "sqlite":
                raise NotImplementedError("SQLite logger requires additional parameters. Use SQLiteLogger directly.")
            else:
                raise ValueError(f"Unknown logger type: {spec}")
        if len(loggers) == 1:
            return loggers[0]
        return MultiLogger(*loggers)

    @property
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

    @property
    def ppid(self):
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
        return ReplayEpisode(self.rundir, time_step, test_num, episode, frames, detailed_actions, test_env.action_space, agent)
