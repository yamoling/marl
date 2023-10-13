import json
import os
import shutil
import time
import pickle
from copy import deepcopy
from dataclasses import dataclass
from serde.json import to_json
from typing import Literal, Optional


import polars as pl
from rlenv.models import EpisodeBuilder, RLEnv, Transition

from marl import logging
from marl.qlearning import IDeepQLearning
from marl.utils import encode_b64_image, exceptions, stats

from .algo import RLAlgo
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run
from .runner import Runner


@dataclass
class Dataset:
    label: str
    mean: list[float]
    min: list[float]
    max: list[float]
    std: list[float]
    ci95: list[float]


@dataclass
class Experiment:
    logdir: str
    algo: RLAlgo
    env: RLEnv
    test_interval: int
    n_steps: int
    creation_timestamp: int
    runs: list[Run]
    

    def __init__(self, logdir: str, algo: RLAlgo, env: RLEnv, test_interval: int, n_steps: int, creation_timestamp: int):
        self.logdir = logdir
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
                env=env,
                n_steps=n_steps,
                test_interval=test_interval,
                creation_timestamp=int(time.time() * 1000)
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

    # def as_dict(self) -> dict[str, Any]:
    #     return {
    #         "algorithm": self.algo.name,
    #         "env": {"name": self.env.name, "action_meanings": self.env.action_meanings},
    #         "logdir": self.logdir,
    #         "n_steps": self.n_steps,
    #         "test_interval": self.test_interval,
    #         "creation_timestamp": self.creation_timestamp,
    #         "runs": [run.to_json() for run in self.runs],
    #     }

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
            loggers = "csv", 
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

        runner = Runner(
            env=deepcopy(self.env), 
            algo=deepcopy(self.algo), 
            logger=logger, 
            test_interval=self.test_interval, 
            n_steps=self.n_steps, 
            test_env=deepcopy(self.env)
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
        raise NotImplementedError()
        run = Run.load(rundir)
        if run.is_running:
            raise ValueError(f"{rundir} is already running.")
        runner = self.create_runner(
            "csv",
            "web",
            "tensorboard",
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
    def _metrics(dfs: list[pl.DataFrame]) -> tuple[list[int], list[Dataset]]:
        dfs = [d for d in dfs if not d.is_empty()]
        if len(dfs) == 0:
            return [], []
        df = pl.concat(dfs)
        df = df.drop("timestamp_sec")
        df_stats = stats.stats_by("time_step", df, len(dfs))
        res = []
        for col in df.columns:
            if col == "time_step":
                continue
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
            return self._metrics([run.test_metrics for run in self.runs])
        except ValueError:
            return [], []

    def train_metrics(self):
        try:
            # Round the time step to match the closest test interval
            dfs = [
                stats.round_col(run.train_metrics, "time_step", self.test_interval) for run in self.runs if not run.train_metrics.is_empty()
            ]
            ticks, train_metrics = self._metrics(dfs)

            dfs = [
                stats.round_col(run.training_data, "time_step", self.test_interval) for run in self.runs if not run.training_data.is_empty()
            ]
            _, training_data = self._metrics(dfs)
            return ticks, train_metrics + training_data
        except ValueError:
            return [], []

    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        summary = []
        for run in self.runs:
            summary += run.get_test_episodes(time_step)
        return summary

    def replay_episode(self, episode_folder: str) -> ReplayEpisode:
        # Actions must be loaded because of the stochasticity of the policy
        with open(os.path.join(episode_folder, "actions.json"), "r") as a:
            actions = json.load(a)
        self.algo = pickle.load(open(os.path.join(episode_folder, "algo.pkl"), "rb"))
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
