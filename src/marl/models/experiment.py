import os
import pathlib
import pickle
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import orjson
import torch
from marlenv.models import ActionSpace, MARLEnv
from tqdm import tqdm

from marl import exceptions
from marl.agents import DQN, Agent, ContinuousAgent
from marl.training import NoTrain
from marl.utils import default_serialization, encode_b64_image, stats
from marl.utils.gpu import get_device

from .batch import TransitionBatch
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run
from .runner import Runner
from .trainer import Trainer


@dataclass
class Experiment[A, AS: ActionSpace]:
    logdir: str
    agent: Agent
    trainer: Trainer
    env: MARLEnv[A, AS]
    test_interval: int
    n_steps: int
    creation_timestamp: int
    test_env: MARLEnv[A, AS]

    def __init__(
        self,
        logdir: str,
        agent: Agent,
        trainer: Trainer,
        env: MARLEnv[A, AS],
        test_interval: int,
        n_steps: int,
        creation_timestamp: int,
        test_env: MARLEnv[A, AS],
    ):
        self.logdir = logdir
        self.trainer = trainer
        self.agent = agent
        self.env = env
        self.test_interval = test_interval
        self.n_steps = n_steps
        self.creation_timestamp = creation_timestamp
        self.test_env = test_env

    @staticmethod
    def create(
        env: MARLEnv[A, AS],
        n_steps: int,
        logdir: str = "logs/tests",
        trainer: Optional[Trainer] = None,
        agent: Optional[Agent] = None,
        test_interval: int = 0,
        test_env: Optional[MARLEnv[A, AS]] = None,
    ):
        """Create a new experiment."""
        if test_env is not None:
            if not env.has_same_inouts(test_env):
                raise ValueError("The test environment must have the same inputs and outputs as the training environment.")
        else:
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
            if trainer is None:
                trainer = NoTrain(env)
            if agent is None:
                agent = trainer.make_agent()
            experiment = Experiment(
                logdir,
                agent=agent,
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
            return orjson.loads(f.read())

    def move(self, new_logdir: str):
        """Move an experiment to a new directory."""
        shutil.move(self.logdir, new_logdir)
        self.logdir = new_logdir
        self.save()

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

        with open(os.path.join(self.logdir, "experiment.json"), "wb") as f:
            f.write(orjson.dumps(self, default=default_serialization, option=orjson.OPT_SERIALIZE_NUMPY))
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
        seed: int = 0,
        fill_strategy: Literal["scatter", "group"] = "scatter",
        required_memory_MB: int = 0,
        quiet: bool = False,
        device: Literal["cpu", "auto"] | int = "auto",
        n_tests: int = 1,
        render_tests: bool = False,
    ):
        """Train the Agent on the environment according to the experiment parameters."""
        runner = self.create_runner()
        selected_device = get_device(device, fill_strategy, required_memory_MB)
        runner = runner.to(selected_device)
        runner.run(
            self.logdir,
            seed=seed,
            n_tests=n_tests,
            quiet=quiet,
            n_steps=self.n_steps,
            test_interval=self.test_interval,
            render_tests=render_tests,
        )

    def test_on_other_env(
        self,
        other_env: MARLEnv[A, AS],
        new_logdir: str,
        n_tests: int,
        quiet: bool = False,
        device: Literal["auto", "cpu"] = "auto",
    ):
        """
        Test the Agent on an other environment but with the same parameters.

        This methods loads the experiment parameters at every test step and run the test on the given environment.
        """
        new_experiment = Experiment.create(
            logdir=new_logdir,
            env=deepcopy(self.env),
            n_steps=self.n_steps,
            agent=deepcopy(self.agent),
            trainer=self.trainer,
            test_interval=self.test_interval,
            test_env=other_env,
        )
        runner = new_experiment.create_runner().to(device)
        runs = sorted(list(self.runs), key=lambda run: run.rundir)
        for i, base_run in enumerate(runs):
            new_run = Run.create(new_experiment.logdir, base_run.seed)
            with new_run as run_handle:
                for time_step in tqdm(range(0, base_run.latest_time_step + 1, self.test_interval), desc=f"Run {i}", disable=quiet):
                    self.agent.load(base_run.get_saved_algo_dir(time_step))
                    runner._test_and_log(n_tests, time_step, run_handle=run_handle, quiet=True, render=False)

    def create_runner(self):
        return Runner(
            env=self.env,
            agent=self.agent,
            trainer=self.trainer,
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

    def replay_episode(self, episode_folder: str):
        # Episode folder should look like logs/experiment/run_2021-09-14_14:00:00.000000_seed=0/test/<time_step>/<test_num>
        # possibly with a trailing slash
        path = pathlib.Path(episode_folder)
        test_num = int(path.name)
        time_step = int(path.parent.name)
        run = Run.load(path.parent.parent.parent.as_posix())
        self.agent.load(run.get_saved_algo_dir(time_step))
        runner = self.create_runner()
        seed = runner.get_test_seed(time_step, test_num)
        actions = run.get_test_actions(time_step, test_num)
        episode = self.test_env.replay(actions, seed=seed)  # type: ignore
        # episode = runner.test(seed)
        frames = [encode_b64_image(img) for img in episode.get_images(self.test_env, seed=seed)]
        replay = ReplayEpisode(episode_folder, episode, frames)

        # Add extra data to the replay depending on the algorithm
        obs = torch.from_numpy(np.array(episode.obs))
        extras = torch.from_numpy(np.array(episode.extras))
        actions = torch.from_numpy(np.array(episode.actions))
        if isinstance(self.agent, ContinuousAgent):
            dist = self.agent.actor_network.policy(obs, extras)
            logits = dist.log_prob(actions)
            replay.logits = logits.tolist()
            replay.probs = torch.exp(logits).tolist()
            replay.state_values = self.agent.actor_network.value(obs, extras).tolist()
        elif isinstance(self.agent, DQN):
            batch = TransitionBatch(list(episode.transitions()))
            replay.qvalues = self.agent.qnetwork.batch_forward(batch.obs, batch.extras).detach().cpu().tolist()
        return replay
