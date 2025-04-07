from copy import deepcopy
from pprint import pprint
from typing import Literal, Optional

import numpy as np
import torch
from marlenv import ActionSpace, Episode, MARLEnv, Transition
from tqdm import tqdm
from typing_extensions import TypeVar

from marl.agents import Agent
from marl.agents.random_agent import RandomAgent
from marl.models.run import Run, RunHandle
from marl.models.trainer import Trainer
from marl.training import NoTrain
from marl.utils import get_device

A = TypeVar("A", bound=ActionSpace)


class Runner[A, AS: ActionSpace]:
    _env: MARLEnv[A, AS]
    _agent: Agent
    _trainer: Trainer
    _test_env: MARLEnv[A, AS]

    def __init__(
        self,
        env: MARLEnv[A, AS],
        agent: Optional[Agent] = None,
        trainer: Optional[Trainer] = None,
        test_env: Optional[MARLEnv[A, AS]] = None,
    ):
        self._trainer = trainer or NoTrain(env)
        self._env = env
        if agent is None:
            agent = RandomAgent(env)
        self._agent = agent  #  or RandomAlgo(env)
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env

    def _train_episode(
        self,
        step_num: int,
        episode_num: int,
        n_tests: int,
        quiet: bool,
        run_handle: RunHandle,
        max_step: int,
        test_interval: int,
        render_tests: bool,
    ):
        obs, state = self._env.reset()
        self._agent.new_episode()
        episode = Episode.new(obs, state, metrics={"initial_value": self._agent.value(obs)})
        while not episode.is_finished and step_num < max_step:
            if n_tests > 0 and test_interval > 0 and step_num % test_interval == 0:
                self._test_and_log(n_tests, step_num, quiet, run_handle, render_tests)
            match self._agent.choose_action(obs):
                case (action, qvalues, dict(kwargs)):
                    step = self._env.step(action)
                case (action, qvalues):
                    step = self._env.step(action)
                    kwargs = {}
            if step_num == max_step:
                step.truncated = True
            transition = Transition.from_step(obs, state, action, qvalues, step, **kwargs)
            training_metrics = self._trainer.update_step(transition, step_num)
            run_handle.log_train_step(training_metrics, step_num)
            episode.add(transition)
            obs = step.obs
            state = step.state
            step_num += 1
        training_logs = self._trainer.update_episode(episode, episode_num, step_num)
        run_handle.log_train_episode(episode, step_num, training_logs)
        return episode

    def run(
        self,
        logdir: str,
        seed: int = 0,
        n_tests: int = 1,
        n_steps: int = 1_000_000,
        test_interval: int = 5_000,
        quiet: bool = False,
        render_tests: bool = False,
    ):
        """Start the training loop"""
        import random

        import torch

        # The test environment is seeded at each testing step for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._env.seed(seed)
        self._agent.randomize()
        self._trainer.randomize()

        max_step = n_steps
        episode_num = 0
        step = 0
        pbar = tqdm(total=n_steps, desc="Training", unit="Step", leave=True, disable=quiet)
        with Run.create(logdir, seed) as run:
            while step < max_step:
                episode = self._train_episode(
                    step_num=step,
                    episode_num=episode_num,
                    n_tests=n_tests,
                    quiet=quiet,
                    run_handle=run,
                    max_step=max_step,
                    test_interval=test_interval,
                    render_tests=render_tests,
                )
                episode_num += 1
                step += len(episode)
                pbar.update(len(episode))
            # Test the final agent
            if n_tests > 0 and test_interval > 0:
                self._test_and_log(n_tests, n_steps, quiet, run, render_tests)
        pbar.close()

    def _test_and_log(self, n_tests: int, time_step: int, quiet: bool, run_handle: RunHandle, render: bool):
        run_handle.save_agent(self._agent, time_step)
        episodes = self.tests(n_tests, time_step, quiet, render)
        run_handle.log_tests(episodes, time_step)
        self._agent.set_training()

    @staticmethod
    def get_test_seed(time_step: int, test_num: int):
        return time_step + test_num

    def perform_one_test(self, seed: Optional[int] = None, render: bool = False):
        """
        Perform a single test episode.

        The test can be seeded for reproducibility purposes, for instance when the policy or the environment is stochastic.
        """
        self._agent.set_testing()
        if seed is not None:
            self._test_env.seed(seed)
            self._agent.seed(seed)
        self._agent.new_episode()
        obs, state = self._test_env.reset()
        episode = Episode.new(obs, state)
        episode.add_metrics({"initial_value": self._agent.value(obs)})
        i = 0
        while not episode.is_finished:
            i += 1
            if render:
                self._test_env.render()
            match self._agent.choose_action(obs):
                case (action, qvalues, _):
                    step = self._test_env.step(action)
                case (action, qvalues):
                    step = self._test_env.step(action)
            transition = Transition.from_step(obs, state, action, qvalues, step)
            episode.add(transition)
            obs = step.obs
            state = step.state
        if render:
            self._test_env.render()
        return episode

    def tests(self, n_tests: int, time_step: int, quiet: bool = True, render: bool = False):
        """Test the agent"""
        episodes = list[Episode[A]]()
        for test_num in tqdm(range(n_tests), desc="Testing", unit="Episode", leave=True, disable=quiet):
            seed = self.get_test_seed(time_step, test_num)
            episodes.append(self.perform_one_test(seed, render))
        if not quiet:
            metrics = episodes[0].metrics.keys()
            avg_metrics = {m: sum([e.metrics[m] for e in episodes]) / n_tests for m in metrics}
            pprint(avg_metrics)
        return episodes

    def to(self, device: Literal["auto", "cpu"] | int | torch.device):
        match device:
            case str():
                device = get_device(device)
            case int():
                device = torch.device(device)
        self._agent.to(device)
        self._trainer.to(device)
        return self
