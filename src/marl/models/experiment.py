import os
import json
import time
import shutil
import polars as pl
from dataclasses import dataclass
from copy import deepcopy
from typing import Literal, Optional, Any

import rlenv
from rlenv.models import RLEnv, EpisodeBuilder, Transition
import laser_env
from marl import logging
from marl.qlearning import IDeepQLearning
from marl.utils import encode_b64_image, defaults_to, exceptions
from marl.utils import stats

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
    ci95: list[float]

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "ci95": self.ci95,
        }


@dataclass
class Experiment:
    logdir: str
    runs: list[Run]
    algo: RLAlgo
    env: RLEnv
    test_env: RLEnv
    test_interval: int
    s_steps: int

    def __init__(
        self,
        logdir: str,
        algo: RLAlgo,
        env: RLEnv,
        test_env: RLEnv,
        test_interval: int,
        n_steps: int,
    ):
        """This constructor should not be called directly. Use Experiment.create() or Experiment.load() instead."""
        self.runs = []
        for run in os.listdir(logdir):
            if run.startswith("run_"):
                try:
                    self.runs.append(Run.load(os.path.join(logdir, run)))
                except Exception:
                    pass

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
        test_interval: int,
        test_env: Optional[RLEnv] = None,
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
            test_env = defaults_to(test_env, lambda: deepcopy(env))
            experiment = Experiment(
                logdir,
                algo=algo,
                env=env,
                n_steps=n_steps,
                test_env=test_env,
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
        try:
            import marl

            with open(os.path.join(logdir, "experiment.json"), "r", encoding="utf-8") as f:
                summary = json.load(f)
            algo = marl.from_summary(summary["algorithm"])
            env = rlenv.from_summary(summary["env"])
            test_env = rlenv.from_summary(summary["test_env"])
            return Experiment(
                logdir=logdir,
                algo=algo,
                env=env,
                test_env=test_env,
                test_interval=summary["test_interval"],
                n_steps=summary["n_steps"],
            )
        except exceptions.MissingParameterException as e:
            raise exceptions.CorruptExperimentException(f"\n\tUnable to load experiment from {logdir}:{e}")
        except json.decoder.JSONDecodeError:
            raise exceptions.CorruptExperimentException(
                f"\n\tUnable to load experiment from {logdir}: experiment.json is not a valid JSON file."
            )
        except KeyError:
            raise exceptions.CorruptExperimentException(
                f"\n\tUnable to load experiment from {logdir}: experiment.json is missing some fields."
            )

    @staticmethod
    def is_experiment_directory(logdir: str) -> bool:
        """Check if a directory is an experiment directory."""
        try:
            return os.path.exists(os.path.join(logdir, "experiment.json"))
        except:
            return False

    @staticmethod
    def find_experiment_directory(subdir: str) -> str | None:
        """Find the experiment directory containing a given subdirectory."""
        if Experiment.is_experiment_directory(subdir):
            return subdir
        parent = os.path.dirname(subdir)
        if parent == subdir:
            return None
        return Experiment.find_experiment_directory(parent)

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": self.algo.summary(),
            "env": {**self.env.summary(), "action_meanings": self.env.action_meanings},
            "test_env": self.test_env.summary(),
            "logdir": self.logdir,
            "n_steps": self.n_steps,
            "timestamp_ms": int(time.time() * 1000),
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
        rundir: Optional[str] = None,
        quiet=True,
    ) -> Runner:
        if rundir is None:
            rundir = os.path.join(self.logdir, f"run_{time.time()}")
            os.makedirs(rundir, exist_ok=False)
        if len(loggers) == 0:
            loggers = ("web", "tensorboard", "csv")
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
        algo.logger = logger
        env = deepcopy(self.env)

        runner = Runner(
            env=env,
            algo=algo,
            logger=logger,
            test_interval=self.test_interval,
            n_steps=self.n_steps,
            test_env=self.test_env,
        )
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
            raise ValueError(f"{rundir} is already running.")
        runner = self.create_runner(
            "csv",
            "web",
            "tensorboard",
            forced_rundir=rundir,
        )
        runner._algo.load(run.latest_checkpoint)
        try:
            shutil.copytree(run.latest_checkpoint, rundir)
        except FileExistsError:
            pass
        runner._start_step = run.current_step
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
        # df = df.drop("timestamp_sec")
        df_stats = stats.stats_by("time_step", df)
        res = []
        columns = [col for col in df.columns if col not in ["time_step", "timestamp_sec"]]
        if "in_elevator" in columns:
            columns.append("exit_rate")
        for col in columns:
            res.append(
                Dataset(
                    label=col,
                    mean=df_stats[f"mean_{col}"].to_list(),
                    std=df_stats[f"std_{col}"].to_list(),
                    min=df_stats[f"min_{col}"].to_list(),
                    max=df_stats[f"max_{col}"].to_list(),
                    ci95=df_stats[f"ci95_{col}"].to_list(),
                )
            )
        return df_stats["time_step"].to_list(), res

    def test_metrics(self) -> tuple[list[int], list[Dataset]]:
        try:
            dfs = [run.test_metrics for run in self.runs if not run.test_metrics.is_empty()]
            dfs = [df.unique(subset=["time_step"]) for df in dfs]
            df = pl.concat(dfs)
            # df = pl.concat(run.test_metrics.unique() for run in self.runs if not run.test_metrics.is_empty())
            return self._metrics(df)
        except ValueError:
            return [], []

    def train_metrics(self):
        try:
            dfs = [run.train_metrics for run in self.runs if not run.train_metrics.is_empty()]
            # Round the time step to match the closest test interval
            dfs = [stats.round_col(df, "time_step", self.test_interval) for df in dfs]
            # Compute the mean of the metrics for each time step
            dfs = [df.group_by("time_step").mean() for df in dfs]
            df = pl.concat(dfs)
            # Then finally get the metrics averag
            ticks, train_metrics = self._metrics(df)

            df = pl.concat(run.training_data for run in self.runs if not run.training_data.is_empty())
            df = stats.round_col(df, "time_step", self.test_interval)
            _, training_data = self._metrics(df)
            return ticks, train_metrics + training_data
        except ValueError:
            return [], []

    # def training_data(self):
    #     try:
    #         df = pl.concat(run.training_data for run in self.runs if not run.training_data.is_empty())
    #         df = stats.round_col(df, "time_step", self.test_interval)
    #         return self._metrics(df)
    #     except ValueError:
    #         return {}

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
        env = rlenv.from_summary(env_summary)
        obs = env.reset()
        values = []
        frames = [encode_b64_image(env.render("rgb_array"))]
        episode = EpisodeBuilder()
        qvalues = []
        for action in actions:
            values.append(self.algo.value(obs))
            if isinstance(self.algo, IDeepQLearning):
                qvalues.append(self.algo.compute_qvalues(obs).tolist())
            obs_, reward, done, truncated, info = env.step(action)
            episode.add(Transition(obs, action, reward, done, info, obs_, truncated))
            frames.append(encode_b64_image(env.render("rgb_array")))
            obs = obs_
        episode = episode.build()
        return ReplayEpisode(
            directory=episode_folder,
            episode=episode,
            qvalues=qvalues,
            frames=frames,
            metrics=episode.metrics,
            state_values=values,
        )


def restore_env(env_summary: dict[str, Any], force_static=False) -> RLEnv:
    if force_static:
        try:
            return laser_env.StaticLaserEnv.from_summary(env_summary)
        except KeyError:
            return rlenv.from_summary(env_summary)
    return rlenv.from_summary(env_summary)
