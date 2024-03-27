import json
import pathlib
import os
import shutil
import time
import pickle
from typing import Literal, Optional
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from serde.json import to_json
from serde import serde


from rlenv.models import EpisodeBuilder, RLEnv, Transition
from marl.utils import encode_b64_image, exceptions, stats

from .algo import RLAlgo
from .trainer import Trainer
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run
from .runner import Runner


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
    test_env: RLEnv

    def __init__(
        self,
        logdir: str,
        algo: RLAlgo,
        trainer: Trainer,
        env: RLEnv,
        test_interval: int,
        n_steps: int,
        creation_timestamp: int,
        test_env: RLEnv,
    ):
        self.logdir = logdir
        self.trainer = trainer
        self.algo = algo
        self.env = env
        self.test_interval = test_interval
        self.n_steps = n_steps
        self.creation_timestamp = creation_timestamp
        self.test_env = test_env

    @staticmethod
    def create(
        logdir: str,
        algo: RLAlgo,
        trainer: Trainer,
        env: RLEnv,
        n_steps: int,
        test_interval: int,
        test_env: Optional[RLEnv] = None,
    ) -> "Experiment":
        """Create a new experiment."""
        if test_env is None:
            test_env = deepcopy(env)
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
                test_env=test_env,
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
        return experiment

    @staticmethod
    def get_parameters(logdir: str) -> dict:
        """Get the parameters of an experiment."""
        with open(os.path.join(logdir, "experiment.json"), "r") as f:
            return json.load(f)

    def copy(self, new_logdir: str, copy_runs: bool = True):
        new_exp = deepcopy(self)
        new_exp.logdir = new_logdir
        new_exp.save()
        if copy_runs:
            for run in self.runs:
                new_rundir = run.rundir.replace(self.logdir, new_logdir)
                shutil.copytree(run.rundir, new_rundir)
        return new_exp

    def delete(self):
        shutil.rmtree(self.logdir)

    @property
    def is_running(self):
        """Check if an experiment is running."""
        for run in self.runs:
            if run.is_running:
                return True
        return False

    def get_tests_at(self, time_step: int):
        summary = list[ReplayEpisodeSummary]()
        for run in self.runs:
            summary += run.get_test_episodes(time_step)
        return summary

    def save(self):
        """Save the experiment to disk."""
        os.makedirs(self.logdir, exist_ok=True)
        with open(os.path.join(self.logdir, "experiment.json"), "w") as f:
            f.write(to_json(self))
        with open(os.path.join(self.logdir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

    @property
    def runs(self):
        for run in os.listdir(self.logdir):
            if run.startswith("run_"):
                try:
                    yield Run.load(os.path.join(self.logdir, run))
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

    def run(
        self,
        seed: int,
        quiet: bool = False,
        n_tests: int = 1,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        run_in_new_process=False,
    ):
        if run_in_new_process:
            # Parent process returns directly
            if os.fork() != 0:
                return
        runner = self.create_runner().to(device)
        runner.train(self.logdir, seed, n_tests, quiet)
        if run_in_new_process:
            exit(0)

    def create_runner(self):
        return Runner(
            env=self.env,
            algo=self.algo,
            trainer=self.trainer,
            test_interval=self.test_interval,
            n_steps=self.n_steps,
            test_env=self.test_env,
        )

    @property
    def train_dir(self):
        return os.path.join(self.logdir, "train")

    @property
    def test_dir(self):
        return os.path.join(self.logdir, "test")

    def n_active_runs(self):
        return len([run for run in self.runs if run.is_running])

    def get_experiment_results(self, replace_inf=False):
        """Get all datasets of an experiment."""
        runs = list(self.runs)
        datasets = stats.compute_datasets([run.test_metrics for run in runs], self.logdir, replace_inf, suffix=" [test]")
        datasets += stats.compute_datasets(
            [run.train_metrics(self.test_interval) for run in runs], self.logdir, replace_inf, suffix=" [train]"
        )
        datasets += stats.compute_datasets([run.training_data(self.test_interval) for run in runs], self.logdir, replace_inf)
        return datasets

    def replay_episode(self, episode_folder: str) -> ReplayEpisode:
        # Episode folder should look like logs/experiment/run_2021-09-14_14:00:00.000000_seed=0/test/<time_step>/<test_num>
        # possibly with a trailing slash
        path = pathlib.Path(episode_folder)
        test_num = int(path.name)
        time_step = int(path.parent.name)
        rundir = path.parent.parent.parent
        run = Run.load(rundir.as_posix())
        actions = run.get_test_actions(time_step, test_num)
        self.algo.load(run.get_saved_algo_dir(time_step))
        try:
            env = run.get_test_env(time_step)
        except exceptions.TestEnvNotSavedException:
            # The environment has not been saved, fallback to the local one
            env = deepcopy(self.env)

        env.seed(time_step + test_num)
        obs = env.reset()
        values = []
        frames = [encode_b64_image(env.render("rgb_array"))]
        episode = EpisodeBuilder()
        try:
            for action in actions:
                values.append(self.algo.value(obs))
                obs_, reward, done, truncated, info = env.step(action)
                episode.add(Transition(obs, np.array(action), reward, done, info, obs_, truncated))
                frames.append(encode_b64_image(env.render("rgb_array")))
                obs = obs_
            episode = episode.build()
            from marl.qlearning import DQN
            from marl.models.batch import TransitionBatch

            if isinstance(self.algo, DQN):
                batch = TransitionBatch(list(episode.transitions()))
                qvalues = self.algo.qnetwork.batch_forward(batch.obs, batch.extras)
                qvalues = qvalues.cpu().detach().tolist()
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
                "Not possible to replay the episode. Maybe the enivornment state was not saved properly or does not support (un)pickling."
            )
