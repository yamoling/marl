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
from marlenv.models import MARLEnv
from tqdm import tqdm

from marl import exceptions
from marl.agents import DQN, Agent, ContinuousAgent
from marl.models.run import Run
from marl.models.runner import Runner
from marl.models.trainer import Trainer
from marl.models.batch import TransitionBatch
from marl.models.replay_episode import ReplayEpisode
from marl.training import NoTrain
from marl.utils import encode_b64_image
from marl.utils.gpu import get_device


from .light_experiment import LightExperiment


@dataclass
class Experiment[A](LightExperiment):
    logdir: str
    agent: Agent
    trainer: Trainer
    env: MARLEnv[A]
    test_interval: int
    n_steps: int
    creation_timestamp: int
    test_env: MARLEnv[A]
    log_qvalues: Optional[bool] = False

    def __init__(
        self,
        logdir: str,
        agent: Agent,
        trainer: Trainer,
        env: MARLEnv[A],
        test_interval: int,
        n_steps: int,
        creation_timestamp: int,
        test_env: MARLEnv[A],
        log_qvalues: Optional[bool],
    ):
        super().__init__(logdir, test_interval, n_steps, creation_timestamp)
        self.trainer = trainer
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.log_qvalues = log_qvalues

    @staticmethod
    def create(
        env: MARLEnv[A],
        n_steps: int,
        logdir: str = "logs/tests",
        trainer: Optional[Trainer] = None,
        agent: Optional[Agent] = None,
        test_interval: int = 0,
        test_env: Optional[MARLEnv[A]] = None,
        log_qvalues: Optional[bool] = False,
    ):
        """Create a new experiment."""
        if test_env is not None:
            if not env.has_same_inouts(test_env):
                raise ValueError("The test environment must have the same inputs and outputs as the training environment.")
        else:
            test_env = deepcopy(env)

        if not logdir.startswith(os.path.join("logs", "")):
            logdir = os.path.join("logs", logdir)

            # Remove the test and debug logs
        if logdir in [os.path.join("logs", "test"), os.path.join("logs", "debug"), os.path.join("logs", "tests")]:
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
                log_qvalues=log_qvalues,
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

    def save(self):
        """Save the experiment to disk."""
        super().save()
        with open(os.path.join(self.logdir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

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
        if device != "cpu" and device != "auto": device = int(device)
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
        other_env: MARLEnv[A],
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
            log_qvalues=self.log_qvalues,
        )

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
