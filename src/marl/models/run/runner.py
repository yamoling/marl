import os
from copy import deepcopy
from pprint import pprint
from typing import Literal, Optional

import torch
from marlenv import ActionSpace, Episode, MARLEnv, Transition
from tqdm import tqdm
from typing_extensions import TypeVar

from marl.agents import Agent
from marl.agents.random_agent import RandomAgent
from marl.logging import CSVLogger
from marl.models.trainer import Trainer
from marl.training import NoTrain
from marl.utils import get_device

from .run import Run

A = TypeVar("A", bound=ActionSpace)


class Runner[A, AS: ActionSpace](Run):
    _env: MARLEnv[A, AS]
    _agent: Agent
    _trainer: Trainer
    _test_env: MARLEnv[A, AS]

    def __init__(
        self,
        rundir: str,
        seed: int,
        n_tests: int,
        quiet: bool,
        test_interval: int,
        n_steps: int,
        env: MARLEnv[A, AS],
        agent: Optional[Agent] = None,
        trainer: Optional[Trainer] = None,
        test_env: Optional[MARLEnv[A, AS]] = None,
    ):
        self.logger = CSVLogger(rundir)
        super().__init__(rundir, seed, n_tests, test_interval, n_steps, self.logger.reader(rundir))
        if trainer is None:
            trainer = NoTrain(env)
        self._trainer = trainer
        self._env = env
        if agent is None:
            agent = RandomAgent(env)
        self._agent = agent
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env
        self._quiet = quiet

    @staticmethod
    def from_experiment(experiment, seed: int, quiet: bool = False, n_tests: int = 1):
        from marl import Experiment

        # Type hinting
        assert isinstance(experiment, Experiment)
        run = Run.from_experiment(experiment, seed, n_tests=n_tests)
        return Runner.from_run(
            run,
            experiment.env,
            experiment.agent,
            experiment.trainer,
            test_env=experiment.test_env,
            quiet=quiet,
        )

    @staticmethod
    def from_run(
        dead_run: Run,
        env: MARLEnv[A, AS],
        agent: Agent,
        trainer: Trainer,
        quiet: bool = False,
        test_env: Optional[MARLEnv[A, AS]] = None,
    ):
        return Runner(
            rundir=dead_run.rundir,
            seed=dead_run.seed,
            n_tests=dead_run.n_tests,
            quiet=quiet,
            test_interval=dead_run.test_interval,
            n_steps=dead_run.n_steps,
            env=env,
            agent=agent,
            trainer=trainer,
            test_env=test_env,
        )

    def _train_episode(
        self,
        step_num: int,
        episode_num: int,
        render_tests: bool,
    ):
        obs, state = self._env.reset()
        self._agent.new_episode()
        episode = Episode.new(obs, state, metrics={"initial_value": self._agent.value(obs)})
        while not episode.is_finished and step_num < self.n_steps:
            if self.n_tests > 0 and self.test_interval > 0 and step_num % self.test_interval == 0:
                self._test_and_log(step_num, render_tests)
            match self._agent.choose_action(obs):
                case (action, dict(kwargs)):
                    step = self._env.step(action)
                case action:
                    step = self._env.step(action)
                    kwargs = {}
            if step_num == self.n_steps:
                step.truncated = True
            transition = Transition.from_step(obs, state, action, step, **kwargs)
            training_metrics = self._trainer.update_step(transition, step_num)
            self.logger.train.log(training_metrics, step_num)
            episode.add(transition)
            obs = step.obs
            state = step.state
            step_num += 1
        training_logs = self._trainer.update_episode(episode, episode_num, step_num)
        self.logger.train.log(episode.metrics, step_num)
        self.logger.training_data.log(training_logs, step_num)
        return episode

    def run(self, render_tests: bool = False):
        """Start the training loop"""
        import marl

        marl.seed(self.seed, self._env)
        self._agent.randomize()
        self._trainer.randomize()

        with open(self.pid_filename, "w") as f:
            f.write(str(os.getpid()))

        episode_num = 0
        step = 0
        pbar = tqdm(total=self.n_steps, desc="Training", unit="Step", leave=True, disable=self._quiet)
        while step < self.n_steps:
            episode = self._train_episode(step, episode_num, render_tests)
            episode_num += 1
            step += len(episode)
            pbar.update(len(episode))
        # Test the final agent
        if self.n_tests > 0 and self.test_interval > 0:
            self._test_and_log(self.n_steps, render_tests)
        pbar.close()
        self.close()

    def _test_and_log(self, time_step: int, render: bool):
        self._agent.save(self.get_saved_algo_dir(time_step))
        episodes = self.tests(time_step, render)
        self.logger.log_tests(episodes, time_step)
        self._agent.set_training()

    def perform_one_test(self, time_step: int, test_num: int, render: bool = False):
        """
        Perform a single test episode.

        The test can be seeded for reproducibility purposes, for instance when the policy or the environment is stochastic.
        """
        self._agent.set_testing()
        seed = self.get_test_seed(time_step, test_num)
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
                case (action, _):
                    step = self._test_env.step(action)
                case action:
                    step = self._test_env.step(action)
            transition = Transition.from_step(obs, state, action, step)
            episode.add(transition)
            obs = step.obs
            state = step.state
        if render:
            self._test_env.render()
        return episode

    def tests(self, time_step: int, render: bool = False):
        """Test the agent"""
        episodes = list[Episode[A]]()
        for test_num in tqdm(range(self.n_tests), desc="Testing", unit="Episode", leave=True, disable=self._quiet):
            episodes.append(self.perform_one_test(time_step, test_num, render))
        if not self._quiet:
            metrics = episodes[0].metrics.keys()
            avg_metrics = {}
            for key in metrics:
                try:
                    avg_metrics[key] = sum([e.metrics[key] for e in episodes]) / self.n_tests
                except TypeError:
                    pass
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

    def close(self):
        try:
            os.remove(self.pid_filename)
        except FileNotFoundError:
            pass

    def __del__(self):
        self.close()

    def agent_at(self, time_step: int) -> Agent:
        self._agent.load(self.get_saved_algo_dir(time_step))
        return self._agent
