import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import torch
from marlenv import Episode, MARLEnv, Space, Transition
from tqdm import tqdm

from marl.utils import get_device

if TYPE_CHECKING:
    from marl import Agent, Experiment, Run, Trainer
    from marl.logging import Logger


class SimpleRunner[A: Space]:
    """
    The Simple Runner contains the main logic to perform one single run. Every runner should rely on `SimpleRunner` to perform the training loop in order to remain consistent.
    """

    def __init__(
        self,
        n_steps: int,
        env: MARLEnv[A],
        n_tests: int = 1,
        test_interval: int = 5000,
        quiet: bool = False,
        agent: "Agent[npt.ArrayLike] | None" = None,
        trainer: "Trainer[npt.ArrayLike] | None" = None,
        test_env: MARLEnv[A] | None = None,
        save_weights: bool = False,
        save_actions: bool = True,
    ):
        if trainer is None:
            from marl.training import NoTrain

            trainer = NoTrain(env)
        self.n_steps = n_steps
        self.test_interval = test_interval
        self.n_tests = n_tests
        self._trainer = trainer
        self._env = env
        self._save_weights = save_weights
        self._save_actions = save_actions
        if agent is None:
            agent = trainer.make_agent()
        self._agent = agent
        if test_env is None:
            test_env = deepcopy(env)
        self._test_env = test_env
        self._quiet = quiet

    def start(self, run: "Run", render_tests: bool = False):
        """Start the training loop."""
        import marl

        if run.is_running or run.is_completed(self.n_steps):
            return

        marl.seed(run.seed, self._env)
        self._agent.randomize()
        self._trainer.randomize()

        with (
            tqdm(total=self.n_steps, desc="Training", unit="Step", leave=True, disable=self._quiet) as pbar,
            run as logger,
        ):
            episode_num = 0
            step = 0
            while step < self.n_steps:
                episode = self._train_episode(logger, step, episode_num, render_tests)
                episode_num += 1
                step += len(episode)
                pbar.update(len(episode))
            # Test the final agent
            if self.n_tests > 0 and self.test_interval > 0:
                self._test_and_log(logger, self.n_steps, render_tests)

    @staticmethod
    def from_experiment(
        exp: "Experiment[A]",
        n_tests: int = 1,
        quiet: bool = False,
    ):
        return SimpleRunner(
            exp.n_steps,
            exp.env,
            n_tests,
            exp.test_interval,
            quiet,
            exp.trainer.make_agent(),
            exp.trainer,
            exp.test_env,
            save_weights=exp.save_weights,
        )

    def _train_episode(self, logger: "Logger", step_num: int, episode_num: int, render_tests: bool):
        obs, state = self._env.reset()
        self._agent.new_episode()
        episode = Episode.new(obs, state, metrics={"initial_value": self._trainer.value(obs, state), "episode_num": episode_num})
        while not episode.is_finished and step_num < self.n_steps:
            if self.n_tests > 0 and self.test_interval > 0 and step_num % self.test_interval == 0:
                self._test_and_log(logger, step_num, render_tests)
            action = self._agent.choose_action(obs)
            step = self._env.step(action)
            if step_num == self.n_steps:
                step.truncated = True
            transition = Transition.from_step(obs, state, action.action, step, **action.details)
            training_metrics = self._trainer.update_step(transition, step_num)
            logger.log_training_data(training_metrics, step_num)
            episode.add(transition)
            obs = step.obs
            state = step.state
            step_num += 1
        logger.log_train(episode.metrics, step_num)
        training_logs = self._trainer.update_episode(episode, episode_num, step_num)
        logger.log_training_data(training_logs, step_num)
        return episode

    def _test_and_log(self, logger: "Logger", time_step: int, render: bool):
        if self._save_weights:
            logger.save_agent(self._agent, time_step)
        episodes = self.perform_tests(time_step, render)
        logger.log_test_episodes(episodes, time_step, self._save_actions)

    def perform_tests(self, time_step: int, render: bool = False):
        """Test the agent"""
        self._agent.set_testing()
        episodes = list[Episode]()
        for test_num in tqdm(range(self.n_tests), desc="Testing", unit="Episode", leave=True, disable=self._quiet):
            seed = get_test_seed(time_step, test_num)
            episodes.append(seeded_rollout(self._test_env, self._agent, seed, render, compute_frames=False)[0])
        if not self._quiet:
            metrics = episodes[0].metrics.keys()
            avg_metrics = {}
            for key in metrics:
                try:
                    avg_metrics[key] = sum([e.metrics[key] for e in episodes]) / self.n_tests
                except TypeError:
                    pass
            logging.info(avg_metrics)
        self._agent.set_training()
        return episodes

    def to(self, device: Literal["auto", "cpu"] | int | torch.device):
        device = get_device(device)
        self._agent = self._agent.to(device)
        self._trainer = self._trainer.to(device)
        return self


def seeded_rollout(env: MARLEnv, agent: "Agent", seed: int, render=False, compute_frames=False):
    agent.set_testing()
    env.seed(seed)
    agent.seed(seed)

    agent.new_episode()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    frames = list[npt.NDArray[np.uint8]]()
    action_details = []
    while not episode.is_finished:
        if render:
            env.render()
        if compute_frames:
            frames.append(env.get_image())
        action = agent.choose_action(obs, with_details=True)
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


def get_test_seed(time_step: int, test_num: int):
    return time_step * 31 + test_num
