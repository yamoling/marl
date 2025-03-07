import os
import pickle
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, overload
import pathlib

import numpy as np
import torch
from marlenv.models import ActionSpace, MARLEnv
from tqdm import tqdm

from marl import exceptions
from marl.agents import DQN, Agent, ContinuousAgent
from marl.models.run import Run, Runner
from marl.models.trainer import Trainer
from marl.models.batch import TransitionBatch
from marl.models.replay_episode import ReplayEpisode
from marl.training import NoTrain
from marl.utils import encode_b64_image
from marl.utils.gpu import get_device


from .light_experiment import LightExperiment


@dataclass
class Experiment[A, AS: ActionSpace](LightExperiment):
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
        super().__init__(logdir, test_interval, n_steps, creation_timestamp)
        self.trainer = trainer
        self.agent = agent
        self.env = env
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
        if test_env is None:
            test_env = deepcopy(env)
        if not env.has_same_inouts(test_env):
            raise ValueError("The test environment must have the same inputs and outputs as the training environment.")

        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)

        # Remove the test and debug logs
        if logdir in ("logs/test", "logs/debug", "logs/tests"):
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
        runner = Runner.from_experiment(self, seed, quiet=quiet, n_tests=n_tests)
        selected_device = get_device(device, fill_strategy, required_memory_MB)
        runner = runner.to(selected_device)
        runner.run(render_tests)

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
        runs = sorted(list(self.runs), key=lambda run: run.rundir)
        for i, base_run in enumerate(runs):
            runner = Runner.from_experiment(new_experiment, base_run.seed).to(device)
            for time_step in tqdm(range(0, base_run.latest_time_step + 1, self.test_interval), desc=f"Run {i}", disable=quiet):
                self.agent.load(base_run.get_saved_algo_dir(time_step))
                runner._test_and_log(time_step, render=False)

    @overload
    def replay_episode(self, run_num: int, time_step: int, test_num: int, /) -> ReplayEpisode:
        """
        Replay the `test_num`th test episode at the `time_step`th test step from the `run_num`th run. 
        
        Note that the actions are not re-evaluated from the agent but loaded from the `actions.json` file.
        """

    @overload
    def replay_episode(self, episode_folder: str, /) -> ReplayEpisode:
        """Replay the episode whose actions are saved in the given test folder."""

    def replay_episode(self, *args):
        match args:
            case (run_num, time_step, test_num):
                return self._replay_episode(run_num, time_step, test_num)
            case (episode_folder,):
                path = pathlib.Path(episode_folder)
                test_num = int(path.name)
                time_step = int(path.parent.name)
                return self._replay_episode(test_num, time_step, test_num)
            case _:
                raise ValueError("Invalid arguments")

    def _replay_episode(self, run_num: int, time_step: int, test_num: int):
        run = list(self.runs)[run_num]
        episode_folder = run.test_dir(time_step, test_num)
        self.agent.load(run.get_saved_algo_dir(time_step))
        # runner = self.create_runner()
        seed = Run.get_test_seed(time_step, test_num)
        actions = run.get_test_actions(time_step, test_num)
        episode = self.test_env.replay(actions, seed=seed)
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

    def agent_at(self, time_step: int, run_seed: int = 0) -> Agent:
        """Load the agent at a specific time step."""
        if time_step % self.test_interval != 0:
            raise ValueError(f"Time step must be a multiple of the test interval ({self.test_interval})")
        for run in self.runs:
            if run.seed == run_seed:
                self.agent.load(run.get_saved_algo_dir(time_step))
                return self.agent
        raise ValueError(f"No run with seed {run_seed} found in the experiment.")
