import logging
from copy import deepcopy
from pprint import pprint
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Episode, MARLEnv, Space, Transition
from tqdm import tqdm

from marl.agents.random_agent import RandomAgent
from marl.logging import LogSpecs, get_logger
from marl.models.detailed_action import DetailedAction
from marl.models.run import Run
from marl.utils import get_device

if TYPE_CHECKING:
    from marl import Agent, Experiment, Trainer


class SimpleRunner[A: Space]:
    """
    A Simple Runner performs exactly single run.
    """

    def __init__(
        self,
        rundir: str,
        n_steps: int,
        env: MARLEnv[A],
        n_tests: int = 1,
        test_interval: int = 5000,
        quiet: bool = False,
        seed: int = 0,
        agent: "Agent | None" = None,
        trainer: "Trainer|None" = None,
        test_env: MARLEnv[A] | None = None,
        log_type: LogSpecs = "csv",
    ):
        if trainer is None:
            from marl.training import NoTrain

            trainer = NoTrain(env)
        self._logger = get_logger(rundir, log_type)
        self._run = Run(rundir, seed, n_tests, test_interval, n_steps)
        self.n_steps = n_steps
        self.test_interval = test_interval
        self.n_tests = n_tests
        self.seed = seed
        self._trainer = trainer
        self._env = env
        if agent is None:
            try:
                agent = trainer.make_agent()
            except NotImplementedError:
                logging.info("No agent provided, using a random agent")
                agent = RandomAgent(env)
        self._agent = agent
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env
        self._quiet = quiet
        self._logger.log_params(trainer, agent, env, test_env)

    @staticmethod
    def from_experiment(
        experiment: "Experiment",
        seed: int,
        n_tests: int = 1,
        quiet: bool = False,
    ):
        run = Run.from_experiment(experiment, seed, n_tests=n_tests)
        return SimpleRunner.from_run(
            run=run,
            env=experiment.env,
            agent=experiment.trainer.make_agent(),
            trainer=experiment.trainer,
            test_env=experiment.test_env,
            quiet=quiet,
            logger=experiment.logger,
        )

    @staticmethod
    def from_run(
        run: Run,
        env: MARLEnv[A],
        agent: "Agent",
        trainer: "Trainer",
        logger: LogSpecs = "csv",
        quiet: bool = False,
        test_env: Optional[MARLEnv[A]] = None,
    ):
        return SimpleRunner(
            rundir=run.rundir,
            seed=run.seed,
            n_tests=run.n_tests,
            quiet=quiet,
            test_interval=run.test_interval,
            n_steps=run.n_steps,
            env=env,
            agent=agent,
            trainer=trainer,
            test_env=test_env,
            log_type=logger,
        )

    def _train_episode(self, step_num: int, episode_num: int, render_tests: bool):
        obs, state = self._env.reset()
        self._agent.new_episode()
        episode = Episode.new(obs, state, metrics={"initial_value": self._trainer.value(obs, state), "episode_num": episode_num})
        while not episode.is_finished and step_num < self.n_steps:
            if self.n_tests > 0 and self.test_interval > 0 and step_num % self.test_interval == 0:
                self._test_and_log(step_num, render_tests)
            action = self._agent.choose_action(obs)
            step = self._env.step(action)
            if step_num == self.n_steps:
                step.truncated = True
            transition = Transition.from_step(obs, state, action, step)
            training_metrics = self._trainer.update_step(transition, step_num)
            self._logger.log_train(training_metrics, step_num)
            episode.add(transition)
            obs = step.obs
            state = step.state
            step_num += 1
        self._logger.log_train(episode.metrics, step_num)
        training_logs = self._trainer.update_episode(episode, episode_num, step_num)
        self._logger.log_training_data(training_logs, step_num)
        return episode

    def run(self, render_tests: bool = False):
        """Start the training loop"""
        import marl

        marl.seed(self.seed, self._env)
        self._agent.randomize()
        self._trainer.randomize()

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

    def _test_and_log(self, time_step: int, render: bool):
        self._agent.save(self._run.get_saved_algo_dir(time_step))
        self._trainer.save(self._run.get_saved_algo_dir(time_step))
        episodes = self.tests(time_step, render)
        self._logger.log_test_episodes(episodes, time_step)

    def tests(self, time_step: int, render: bool = False):
        """Test the agent"""
        episodes = list[Episode]()
        for test_num in tqdm(range(self.n_tests), desc="Testing", unit="Episode", leave=True, disable=self._quiet):
            seed = self._run.get_test_seed(time_step, test_num)
            episodes.append(seeded_rollout(self._test_env, self._agent, seed, render, compute_frames=False)[0])
        if not self._quiet:
            metrics = episodes[0].metrics.keys()
            avg_metrics = {}
            for key in metrics:
                try:
                    avg_metrics[key] = sum([e.metrics[key] for e in episodes]) / self.n_tests
                except TypeError:
                    pass
            pprint(avg_metrics)
        self._agent.set_training()
        return episodes

    def to(self, device: Literal["auto", "cpu"] | int | torch.device):
        device = get_device(device)
        self._agent = self._agent.to(device)
        self._trainer = self._trainer.to(device)
        return self

    def agent_at(self, time_step: int) -> "Agent":
        self._agent.load(self._run.get_saved_algo_dir(time_step))
        return self._agent


def seeded_rollout(env: MARLEnv, agent: "Agent", seed: int, render=False, compute_frames=False):
    agent.set_testing()
    env.seed(seed)
    agent.seed(seed)

    agent.new_episode()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    frames = list[npt.NDArray[np.uint8]]()
    action_details = list[DetailedAction]()
    while not episode.is_finished:
        if render:
            env.render()
        if compute_frames:
            frames.append(env.get_image())
        action = agent.choose_action_with_details(obs)
        action_details.append(action)
        step = env.step(action.action)
        transition = Transition.from_step(obs, state, action.action, step)
        episode.add(transition)
        obs = step.obs
        state = step.state
    if render:
        env.render()
    if compute_frames:
        frames.append(env.get_image())
    return episode, frames, action_details
