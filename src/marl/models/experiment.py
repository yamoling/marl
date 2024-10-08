import json
import os
import pathlib
import pickle
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from marlenv.models import EpisodeBuilder, MARLEnv, Transition
from serde import serde
from serde.json import to_json
from tqdm import tqdm

from marl import exceptions
from marl.algo import DDPG, DQN, PPO, RLAlgo
from ..training.no_train import NoTrain
from ..algo.random_algo import RandomAlgo
from marl.models.nn import MAIC
from marl.training import Trainer
from marl.utils import encode_b64_image, stats
from marl.utils.gpu import get_device

from .batch import TransitionBatch
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run
from .runner import Runner


@serde
@dataclass
class Experiment:
    logdir: str
    algo: RLAlgo
    trainer: Trainer
    env: MARLEnv
    test_interval: int
    n_steps: int
    creation_timestamp: int
    test_env: MARLEnv

    def __init__(
        self,
        logdir: str,
        algo: RLAlgo,
        trainer: Trainer,
        env: MARLEnv,
        test_interval: int,
        n_steps: int,
        creation_timestamp: int,
        test_env: MARLEnv,
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
        env: MARLEnv,
        n_steps: int,
        algo: Optional[RLAlgo] = None,
        trainer: Optional[Trainer] = None,
        test_interval: int = 0,
        test_env: Optional[MARLEnv] = None,
    ) -> "Experiment":
        """Create a new experiment."""
        if test_env is not None:
            MARLEnv.assert_same_inouts(env, test_env)
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
            experiment = Experiment(
                logdir,
                algo=algo or RandomAlgo(env),
                trainer=trainer or NoTrain(),
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
        seed: int = 0,
        fill_strategy: Literal["scatter", "group"] = "scatter",
        required_memory_MB: int = 0,
        quiet: bool = False,
        device: Literal["cpu", "auto"] | int = "auto",
        n_tests: int = 1,
    ):
        """Train the RLAlgo on the environment according to the experiment parameters."""
        runner = self.create_runner().to(get_device(device, fill_strategy, required_memory_MB))
        runner.run(
            self.logdir,
            seed=seed,
            n_tests=n_tests,
            quiet=quiet,
            n_steps=self.n_steps,
            test_interval=self.test_interval,
        )

    def test_on_other_env(
        self,
        test_env: MARLEnv,
        new_logdir: str,
        n_tests: int,
        quiet: bool = False,
        device: Literal["auto", "cpu"] = "auto",
    ):
        """
        Test the RLAlgo on an other environment but with the same parameters.

        This methods loads the experiment parameters at every test step and run the test on the given environment.
        """
        new_experiment = Experiment.create(
            logdir=new_logdir,
            env=self.env,
            n_steps=self.n_steps,
            algo=self.algo,
            trainer=self.trainer,
            test_interval=self.test_interval,
            test_env=test_env,
        )
        runner = new_experiment.create_runner().to(device)
        runs = sorted(list(self.runs), key=lambda run: run.rundir)
        for i, base_run in enumerate(runs):
            new_run = Run.create(new_experiment.logdir, base_run.seed)
            with new_run as run_handle:
                for time_step in tqdm(range(0, base_run.latest_time_step + 1, self.test_interval), desc=f"Run {i}", disable=quiet):
                    self.algo.load(base_run.get_saved_algo_dir(time_step))
                    runner.test(n_tests, time_step, run_handle=run_handle, quiet=True)

    def create_runner(self):
        return Runner(
            env=self.env,
            algo=self.algo,
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
        self.test_env.seed(time_step + test_num)
        obs = self.test_env.reset()
        frames = [encode_b64_image(self.test_env.render("rgb_array"))]
        episode = EpisodeBuilder()
        values = []
        qvalues = []
        llogits = []
        pprobs = []
        messages = []
        received_messages = []
        init_qvalues = []
        self.algo.new_episode()
        self.algo.set_testing()
        for action in actions:
            if isinstance(self.algo, DDPG):
                logits = self.algo.actions_logits(obs)
                dist = torch.distributions.Categorical(logits=logits)
                # probs
                pprobs.append(dist.probs.unsqueeze(-1).tolist())  # type: ignore
                # logits
                logits = self.algo.actions_logits(obs).unsqueeze(-1).tolist()
                logits = [[[-10] if np.isinf(x) else x for x in y] for y in logits]
                llogits.append(logits)

                # state-action value
                probs = dist.probs.unsqueeze(0)  # type: ignore
                state = torch.tensor(obs.state).to(self.algo.device, non_blocking=True).unsqueeze(0)
                value = self.algo.state_action_value(state, probs)  # type: ignore
                values.append(value)
                print(value)

            if isinstance(self.algo, (PPO)):
                logits = self.algo.actions_logits(obs)
                dist = torch.distributions.Categorical(logits=logits)
                # probs
                pprobs.append(dist.probs.unsqueeze(-1).tolist())  # type: ignore

                # logits
                llogits.append(self.algo.actions_logits(obs).unsqueeze(-1).tolist())

                # state value
                value = self.algo.value(obs)
                print(value)

            else:
                values.append(self.algo.value(obs))

            obs_, reward, done, truncated, info = self.test_env.step(action)
            episode.add(Transition(obs, np.array(action), reward, done, info, obs_, truncated))
            frames.append(encode_b64_image(self.test_env.render("rgb_array")))
            obs = obs_
        episode = episode.build()

        if isinstance(self.algo, DQN):
            if isinstance(self.algo.qnetwork, MAIC):
                for transition in episode.transitions():
                    current_qvalues, gated_messages, received_message, current_init_qvalues = self.algo.qnetwork.get_values_and_comms(
                        torch.from_numpy(transition.obs.data), torch.from_numpy(transition.obs.extras)
                    )
                    qvalues.append(current_qvalues.detach().cpu().tolist())
                    if gated_messages is not None and len(received_message) > 0:
                        messages.append(gated_messages.detach().cpu().tolist())
                        received_messages.append(received_message.detach().cpu().tolist())
                        init_qvalues.append(current_init_qvalues.detach().cpu().tolist())
            else:
                batch = TransitionBatch(list(episode.transitions()))
                qvalues = self.algo.qnetwork.batch_forward(batch.obs, batch.extras).detach().cpu().tolist()

        return ReplayEpisode(
            directory=episode_folder,
            episode=episode,
            qvalues=qvalues,
            frames=frames,
            metrics=episode.metrics,
            state_values=values,
            probs=pprobs,
            logits=llogits,
            messages=messages,
            received_messages=received_messages,
            init_qvalues=init_qvalues,
        )
