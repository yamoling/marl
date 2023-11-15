import json
import os
import shutil
import time
import pickle
from copy import deepcopy
from dataclasses import dataclass
from serde.json import to_json
from serde import serde
from typing import Literal


import polars as pl
from rlenv.models import EpisodeBuilder, RLEnv, Transition

from marl import logging
from marl.qlearning import IDeepQLearning
from marl.utils import encode_b64_image, exceptions, stats

from .algo import RLAlgo
from .trainer import Trainer
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run
from .runner import Runner


@serde
@dataclass
class Dataset:
    label: str
    mean: list[float]
    min: list[float]
    max: list[float]
    std: list[float]
    ci95: list[float]


@serde
@dataclass
class ExperimentResults:
    logdir: str
    ticks: list[int]
    train: list[Dataset]
    test: list[Dataset]


@serde
@dataclass
class Experiment:
    logdir: str
    algo: RLAlgo
    trainer: Trainer
    env: RLEnv
    test_interval: int
    n_steps: int
    creation_timestamp: int
    runs: list[Run]

    def __init__(self, logdir: str, algo: RLAlgo, trainer: Trainer, env: RLEnv, test_interval: int, n_steps: int, creation_timestamp: int):
        self.logdir = logdir
        self.trainer = trainer
        self.algo = algo
        self.env = env
        self.test_interval = test_interval
        self.n_steps = n_steps
        self.creation_timestamp = creation_timestamp
        self.runs = []
        self.refresh_runs()

    @staticmethod
    def create(
        logdir: str,
        algo: RLAlgo,
        trainer: Trainer,
        env: RLEnv,
        n_steps: int,
        test_interval: int,
    ) -> "Experiment":
        """Create a new experiment."""
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)

            # Remove the test and debug logs
        if logdir in ["logs/test", "logs/debug", "logs/tests"]:
            try:
                shutil.rmtree(logdir)
            except FileNotFoundError:
                pass
        try:
            os.makedirs(logdir, exist_ok=False)
            experiment = Experiment(
                logdir,
                algo=algo,
                trainer=trainer,
                env=env,
                n_steps=n_steps,
                test_interval=test_interval,
                creation_timestamp=int(time.time() * 1000),
            )
            experiment.save()
            return experiment
        except FileExistsError:
            raise exceptions.ExperimentAlreadyExistsException(logdir)
        except Exception as e:
            # In case the experiment could not be created for another reason, do not create the experiment and remove its directory
            shutil.rmtree(logdir, ignore_errors=True)
            raise e

    @staticmethod
    def load(logdir: str) -> "Experiment":
        """Load an experiment from disk."""
        with open(os.path.join(logdir, "experiment.pkl"), "rb") as f:
            experiment: Experiment = pickle.load(f)
        experiment.refresh_runs()
        return experiment

    @staticmethod
    def get_parameters(logdir: str) -> dict:
        """Get the parameters of an experiment."""
        with open(os.path.join(logdir, "experiment.json"), "r") as f:
            return json.load(f)

    @staticmethod
    def get_runs(logdir: str) -> list[Run]:
        """Get the runs of an experiment."""
        runs = []
        for run in os.listdir(logdir):
            if run.startswith("run_"):
                try:
                    runs.append(Run.load(os.path.join(logdir, run)))
                except Exception:
                    pass
        return runs

    @staticmethod
    def get_tests_at(logdir: str, time_step: int) -> list[ReplayEpisodeSummary]:
        runs = Experiment.get_runs(logdir)
        summary = []
        for run in runs:
            summary += run.get_test_episodes(time_step)
        return summary

    def save(self):
        """Save the experiment to disk."""
        with open(os.path.join(self.logdir, "experiment.json"), "w") as f:
            f.write(to_json(self))
        with open(os.path.join(self.logdir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

    def refresh_runs(self):
        self.runs = []
        for run in os.listdir(self.logdir):
            if run.startswith("run_"):
                try:
                    self.runs.append(Run.load(os.path.join(self.logdir, run)))
                except Exception:
                    pass

    @staticmethod
    def is_experiment_directory(logdir: str) -> bool:
        """Check if a directory is an experiment directory."""
        try:
            return os.path.exists(os.path.join(logdir, "experiment.json"))
        except FileNotFoundError:
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

    def create_runner(
        self,
        *loggers: Literal["web", "tensorboard", "csv"],
        seed: int,
        quiet=True,
    ) -> Runner:
        rundir = os.path.join(self.logdir, f"run_{time.time()}")
        os.makedirs(rundir, exist_ok=False)
        if len(loggers) == 0:
            loggers = ("csv",)
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

        import marl

        marl.seed(seed)
        self.env.seed(seed)
        self.algo.randomize()
        self.trainer.randomize()

        runner = Runner(
            env=self.env,
            algo=self.algo,
            trainer=self.trainer,
            logger=logger,
            test_interval=self.test_interval,
            n_steps=self.n_steps,
            test_env=deepcopy(self.env),
        )
        self.runs.append(Run.create(rundir, seed))
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
    def compute_datasets(dfs: list[pl.DataFrame]) -> tuple[list[int], list[Dataset]]:
        dfs = [d for d in dfs if not d.is_empty()]
        if len(dfs) == 0:
            return [], []
        df = pl.concat(dfs)
        try:
            df = df.drop("timestamp_sec")
        except pl.SchemaFieldNotFoundError:
            pass
        df_stats = stats.stats_by("time_step", df)
        res = []
        for col in df.columns:
            if col == "time_step":
                continue
            res.append(
                Dataset(
                    label=col,
                    mean=df_stats[f"mean_{col}"],
                    std=df_stats[f"std_{col}"],
                    min=df_stats[f"min_{col}"],
                    max=df_stats[f"max_{col}"],
                    ci95=df_stats[f"ci95_{col}"],
                )
            )
        return df_stats["time_step"].to_list(), res

    @staticmethod
    def get_experiment_results(logdir: str) -> ExperimentResults:
        """Get the test metrics of an experiment."""
        runs = Experiment.get_runs(logdir)
        try:
            ticks, test_datasets = Experiment.compute_datasets([run.test_metrics for run in runs])
        except ValueError:
            ticks, test_datasets = [], []
        try:
            dfs = [run.train_metrics for run in runs if not run.train_metrics.is_empty()]
            dfs += [run.training_data for run in runs if not run.training_data.is_empty()]
            if len(ticks) >= 2:
                test_interval = ticks[1] - ticks[0]
                dfs = [stats.round_col(df, "time_step", test_interval) for df in dfs]
            ticks2, train_datasets = Experiment.compute_datasets(dfs)
        except ValueError:
            ticks2, train_datasets = [], []
        if len(ticks) == 0:
            ticks = ticks2
        return ExperimentResults(logdir, ticks, train_datasets, test_datasets)

    def train_metrics(self):
        try:
            # Round the time step to match the closest test interval
            dfs = [
                stats.round_col(run.train_metrics, "time_step", self.test_interval) for run in self.runs if not run.train_metrics.is_empty()
            ]
            ticks, train_metrics = self.compute_datasets(dfs)

            dfs = [
                stats.round_col(run.training_data, "time_step", self.test_interval) for run in self.runs if not run.training_data.is_empty()
            ]
            _, training_data = self.compute_datasets(dfs)
            return ticks, train_metrics + training_data
        except ValueError:
            return [], []

    def replay_episode(self, episode_folder: str) -> ReplayEpisode:
        # Actions must be loaded because of the stochasticity of the policy
        with open(os.path.join(episode_folder, "actions.json"), "r") as a:
            actions = json.load(a)
        self.algo.load(os.path.dirname(episode_folder))
        env: RLEnv = pickle.load(open(os.path.join(episode_folder, "env.pkl"), "rb"))
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
