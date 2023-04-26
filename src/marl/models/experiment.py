import os
import json
import time
import shutil
import polars as pl
from dataclasses import dataclass
from copy import deepcopy
from typing import Literal

import rlenv
from rlenv.models import RLEnv, EpisodeBuilder, Transition
from rlenv import wrappers
import laser_env
from marl import logging
from marl.qlearning import IDeepQLearning
from marl.utils import encode_b64_image, defaults_to, exceptions

from .runner import Runner
from .algo import RLAlgo
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run


@dataclass
class Dataset:
    label: str
    mean: list[float]
    min: list[float]
    max: list[float]
    std: list[float]

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


@dataclass
class Experiment:
    logdir: str
    runs: list[Run]
    algo: RLAlgo
    env: RLEnv
    test_env: RLEnv
    n_steps: int
    test_interval: int

    def __init__(
        self,
        logdir: str,
        algo: RLAlgo,
        env: RLEnv,
        test_env: RLEnv,
        n_steps: int,
        test_interval: int,
    ):
        """This constructor should not be called directly. Use Experiment.create() or Experiment.load() instead."""
        self.runs = [
            Run.load(os.path.join(logdir, run))
            for run in os.listdir(logdir)
            if run.startswith("run_")
        ]
        self.logdir = logdir
        self.algo = algo
        self.env = env
        self.test_env = test_env
        self.n_steps = n_steps
        self.test_interval = test_interval

    @staticmethod
    def create(
        logdir: str,
        algo: RLAlgo,
        env: RLEnv,
        n_steps: int,
        test_interval: int = None,
        test_env: RLEnv = None,
    ) -> "Experiment":
        """Create a new experiment."""
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        try:
            # Remove the test and debug logs
            if logdir in ["logs/test", "logs/debug", "logs/tests"]:
                shutil.rmtree(logdir)
        except FileNotFoundError:
            pass
        try:
            os.makedirs(logdir, exist_ok=False)
            test_interval = defaults_to(test_interval, lambda: n_steps // 100)
            test_env = defaults_to(test_env, lambda: deepcopy(env))
            experiment = Experiment(
                logdir,
                algo=algo,
                env=env,
                test_env=test_env,
                n_steps=n_steps,
                test_interval=test_interval,
            )
            experiment.save()
            return experiment
        except FileExistsError:
            raise exceptions.ExperimentAlreadyExistsException(logdir)
        except Exception as e:
            # In case the experiment could not be created for another reason, remove its directory
            shutil.rmtree(logdir, ignore_errors=True)
            raise e

    @staticmethod
    def load(logdir: str) -> "Experiment":
        """Load an existing experiment."""
        import marl
        try:
            with open(os.path.join(logdir, "experiment.json"), "r", encoding="utf-8") as f:
                summary = json.load(f)
            algo = marl.from_summary(summary["algorithm"])
            env = rlenv.from_summary(summary["env"])
            test_env = rlenv.from_summary(summary["test_env"])
            n_steps = summary["n_steps"]
            test_interval = summary["test_interval"]
            return Experiment(
                logdir,
                algo=algo,
                env=env,
                test_env=test_env,
                n_steps=n_steps,
                test_interval=test_interval,
            )
        except KeyError as e:
            raise exceptions.ExperimentVersionMismatch(e.args[0])

    def summary(self) -> dict[str,]:
        return {
            "algorithm": self.algo.summary(),
            "env": self.env.summary(),
            "test_env": self.test_env.summary(),
            "logdir": self.logdir,
            "timestamp_ms": int(time.time() * 1000),
            "n_steps": self.n_steps,
            "test_interval": self.test_interval,
            "runs": [run.to_json() for run in self.runs],
        }

    def save(self):
        os.makedirs(self.logdir, exist_ok=True)
        with open(os.path.join(self.logdir, "experiment.json"), "w") as f:
            s = self.summary()
            json.dump(s, f)

    def create_runner(
        self,
        *loggers: Literal["web", "tensorboard", "csv"],
        checkpoint: str = None,
        seed: int = None,
        forced_rundir: str = None,
        quiet=True,
    ) -> Runner:
        if forced_rundir is not None:
            rundir = forced_rundir
        else:
            rundir = os.path.join(self.logdir, f"run_{time.time()}")
        os.makedirs(rundir, exist_ok=(checkpoint is not None))
        if len(loggers) == 0:
            loggers = ["web", "tensorboard", "csv"]
        logger_list = []
        for logger in loggers:
            match logger:
                case "web":
                    logger_list.append(logging.WSLogger(rundir))
                case "tensorboard":
                    logger_list.append(logging.TensorBoardLogger(rundir, quiet=quiet))
                case "csv":
                    logger_list.append(logging.CSVLogger(rundir, quiet))
                case other:
                    raise ValueError(f"Unknown logger: {other}")
        logger = logging.MultiLogger(rundir, *logger_list, quiet=quiet)

        algo = deepcopy(self.algo)
        env = deepcopy(self.env)
        if checkpoint is not None:
            algo.load(checkpoint)
            try:
                shutil.copytree(checkpoint, rundir)
            except FileExistsError:
                pass

        runner = Runner(
            env=env,
            algo=algo,
            logger=logger,
            n_steps=self.n_steps,
            start_step=0,
            test_interval=self.test_interval,
        )
        if seed is not None:
            runner.seed(seed)
        self.runs.append(Run.create(rundir))
        return runner

    def stop_runner(self, rundir: str):
        """Stops the runner at the given rundir."""
        for i, run in enumerate(self.runs):
            if run.rundir == rundir:
                run = self.runs.pop(i)
                run.stop()
                return
        # If the run was not found, raise an error
        raise ValueError("This rundir does not exist.")

    def restore_runner(self, rundir: str):
        """Retrieve the runner state and restart it if it is not running"""
        run = Run.load(rundir)
        if run.is_running:
            raise ValueError("This run is already running.")
        runner = self.create_runner(
            "csv",
            "web",
            "tensorboard",
            checkpoint=run.latest_checkpoint,
            forced_rundir=rundir,
        )
        runner._current_step = run.current_step
        return runner

    def delete_run(self, rundir: str):
        for i, run in enumerate(self.runs):
            if run.rundir == rundir:
                run = self.runs.pop(i)
                run.delete()
        # If the run was not found, raise an error
        raise ValueError("This rundir does not exist.")

    @property
    def train_dir(self) -> str:
        return os.path.join(self.logdir, "train")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.logdir, "test")

    @staticmethod
    def _metrics(df: pl.DataFrame) -> tuple[list[int], list[Dataset]]:
        if len(df) == 0:
            return [], []
        df = df.drop("timestamp_sec")
        df_mean = (
            df.groupby("time_step")
            .agg([pl.mean(col) for col in df.columns if col not in ["time_step", "timestamp_sec"]])
            .sort("time_step")
        )
        df_std = (
            df.groupby("time_step")
            .agg([pl.std(col) for col in df.columns if col not in ["time_step", "timestamp_sec"]])
            .sort("time_step")
        )
        df_min = (
            df.groupby("time_step")
            .agg([pl.min(col) for col in df.columns if col not in ["time_step", "timestamp_sec"]])
            .sort("time_step")
        )
        df_max = (
            df.groupby("time_step")
            .agg([pl.max(col) for col in df.columns if col not in ["time_step", "timestamp_sec"]])
            .sort("time_step")
        )
        res = []
        for col in df_mean.columns:
            if col == "time_step":
                continue
            res.append(Dataset(
                label=col, 
                mean=df_mean[col].to_list(), 
                std=df_std[col].to_list(),
                min=df_min[col].to_list(),
                max=df_max[col].to_list(),
            ))
        return df_mean["time_step"].to_list(), res

    def test_metrics(self):
        df = pl.concat(run.test_metrics for run in self.runs)
        return self._metrics(df)

    def train_metrics(self):
        df = pl.concat(run.train_metrics for run in self.runs)
        return self._metrics(df)

    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        summary = []
        for run in self.runs:
            summary += run.get_test_episodes(time_step)
        return summary

    def replay_episode(self, episode_folder: str) -> ReplayEpisode:
        # Actions must be loaded because of the stochasticity of the policy
        with (
            open(os.path.join(episode_folder, "actions.json"), "r") as a,
            open(os.path.join(episode_folder, "env.json"), "r") as e,
        ):
            actions = json.load(a)
            env_summary = json.load(e)

        self.algo.load(os.path.dirname(episode_folder))
        env = restore_env(env_summary, force_static=True)
        obs = env.reset()
        frames = [encode_b64_image(env.render("rgb_array"))]
        episode = EpisodeBuilder()
        qvalues = []
        for action in actions:
            if isinstance(self.algo, IDeepQLearning):
                qvalues.append(self.algo.compute_qvalues(obs).tolist())
            obs_, reward, done, info = env.step(action)
            episode.add(Transition(obs, action, reward, done, info, obs_))
            frames.append(encode_b64_image(env.render("rgb_array")))
            obs = obs_
        episode = episode.build()
        return ReplayEpisode(
            directory=episode_folder,
            episode=episode,
            qvalues=qvalues,
            frames=frames,
            metrics=episode.metrics,
        )


def restore_env(env_summary: dict[str,], force_static=False) -> RLEnv:
    if force_static:
        env = laser_env.StaticLaserEnv.from_summary(env_summary)
        try:
            # Do not restore envPool if force_static
            env_summary["wrappers"].remove("EnvPool")
            env_summary.pop("EnvPool")
        except (KeyError, ValueError):
            pass
    else:
        env = rlenv.from_summary(env_summary)
    return wrappers.from_summary(env, env_summary)
