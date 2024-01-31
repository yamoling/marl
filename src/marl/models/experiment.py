import json
import os
import shutil
import time
from datetime import datetime
import pickle
from copy import deepcopy
from dataclasses import dataclass
from serde.json import to_json
from serde import serde


import polars as pl
from rlenv.models import EpisodeBuilder, RLEnv, Transition

from marl.qlearning import DQN
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
    def get_runs(logdir: str):
        """Get the runs of an experiment."""
        for run in os.listdir(logdir):
            if run.startswith("run_"):
                try:
                    yield Run.load(os.path.join(logdir, run))
                except Exception as e:
                    print(e)

    @staticmethod
    def is_running(logdir: str):
        """Check if an experiment is running."""
        for run in Experiment.get_runs(logdir):
            if run.is_running:
                return True
        return False

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

    def create_runner(self, seed: int) -> Runner:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        rundir = os.path.join(self.logdir, f"run_{now}_seed={seed}")
        os.makedirs(rundir, exist_ok=False)
        import marl

        marl.seed(seed)
        self.env.seed(seed)
        self.algo.randomize()
        self.trainer.randomize()

        return Runner(
            env=self.env,
            algo=self.algo,
            trainer=self.trainer,
            run=Run.create(rundir, seed),
            test_interval=self.test_interval,
            n_steps=self.n_steps,
            test_env=deepcopy(self.env),
        )

    @property
    def train_dir(self) -> str:
        return os.path.join(self.logdir, "train")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.logdir, "test")

    @staticmethod
    def compute_datasets(dfs: list[pl.DataFrame], replace_inf=False) -> tuple[list[int], list[Dataset]]:
        dfs = [d for d in dfs if not d.is_empty()]
        if len(dfs) == 0:
            return [], []
        df = pl.concat(dfs)
        try:
            df = df.drop("timestamp_sec")
        except pl.SchemaFieldNotFoundError:
            pass
        df_stats = stats.stats_by("time_step", df, replace_inf)
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

    @staticmethod
    def get_experiment_results(logdir: str, replace_inf=False) -> ExperimentResults:
        """Get the test metrics of an experiment."""
        runs = list(Experiment.get_runs(logdir))
        try:
            ticks, test_datasets = Experiment.compute_datasets([run.test_metrics for run in runs], replace_inf)
        except ValueError:
            ticks, test_datasets = [], []
        try:
            dfs = [run.train_metrics for run in runs if not run.train_metrics.is_empty()]
            dfs_training_data = [run.training_data for run in runs if not run.training_data.is_empty()]
            if len(ticks) >= 2:
                test_interval = ticks[1] - ticks[0]
                dfs = [stats.round_col(df, "time_step", test_interval) for df in dfs]
                dfs_training_data = [stats.round_col(df, "time_step", test_interval) for df in dfs_training_data]

            ticks2, train_datasets = Experiment.compute_datasets(dfs, replace_inf)
            _, training_data = Experiment.compute_datasets(dfs_training_data, replace_inf)
        except ValueError:
            ticks2, train_datasets, training_data = [], [], []
        if len(ticks) == 0:
            ticks = ticks2
        return ExperimentResults(logdir, ticks, train_datasets + training_data, test_datasets)

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
        try:
            env: RLEnv = pickle.load(open(os.path.join(episode_folder, "env.pkl"), "rb"))
        except FileNotFoundError:
            # The environment has not been saved, so we we the local one
            env = self.env
        obs = env.reset()
        values = []
        frames = [encode_b64_image(env.render("rgb_array"))]
        episode = EpisodeBuilder()
        qvalues = []
        try:
            for action in actions:
                values.append(self.algo.value(obs))
                if isinstance(self.algo, DQN):
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
        except AssertionError:
            raise ValueError(
                "Not possible to replay the episode. Maybe the enivornment state was not saved properly or it does not support (un)pickling."
            )
