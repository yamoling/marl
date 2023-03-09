import json
import os
from copy import deepcopy
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition, Observation
from tqdm import tqdm
from . import logging
from .marl_algo import RLAlgo
from .utils import defaults_to
from marl.models import Experiment

class Runner:
    def __init__(
        self,
        env: RLEnv,
        algo: RLAlgo,
        logger: logging.Logger=None,
        test_env: RLEnv=None
    ):
        self._env = env
        self._test_env = defaults_to(test_env, deepcopy(env))
        self._algo = algo
        if logger is None:
            self._logger = logging.default()
        else:
            self._logger = logger
        self._seed = None
        self._best_score = -float("inf")
        self._checkpoint = os.path.join(self.logdir, "checkpoint")
        self._current_step = 0
        self._episode_builder = None
        self._episode_num = 0
        self._obs: Observation|None = None
        self._experiment = Experiment(self.logdir, self.summary())

    def _before_train_episode(self):
        self._episode_builder = EpisodeBuilder()
        self._obs = self._env.reset()
        self._algo.before_train_episode(self._episode_num)
        self._experiment.save_train_env(self._episode_num, self._env.summary())

    def train(self, n_steps: int, test_interval: int=200, n_tests: int=10, quiet=False) -> str:
        """Start the training loop"""
        stop = self._current_step + n_steps
        if self._episode_num == 0:
            self._before_train_episode()
        while self._current_step < stop:
            if self._current_step % test_interval == 0:
                self.test(n_tests, quiet=quiet)
            if self._episode_builder.is_done:
                episode = self._episode_builder.build()
                self._algo.after_train_episode(self._episode_num, episode)
                self._logger.log("Train", episode.metrics, self._current_step)
                self._episode_num += 1
                self._before_train_episode()
            action = self._algo.choose_action(self._obs)
            obs_, reward, done, info = self._env.step(action)
            transition = Transition(self._obs, action, reward, done, info, obs_)
            self._algo.after_step(transition, self._current_step)
            self._episode_builder.add(transition)
            self._obs = obs_
            self._current_step += 1

    def test(self, ntests: int, quiet=False):
        """Test the agent"""
        self._algo.before_tests(self._current_step)
        episodes: list[Episode] = []
        for i in tqdm(range(ntests), desc="Testing", unit="Episode", leave=True, disable=quiet):
            self._algo.before_test_episode(self._current_step, i)
            episode = EpisodeBuilder()
            obs = self._test_env.reset()
            while not episode.is_done:
                action = self._algo.choose_action(obs)
                new_obs, reward, done, info = self._test_env.step(action)
                transition = Transition(obs, action, reward, done, info, new_obs)
                episode.add(transition)
                obs = new_obs
            episode = episode.build()
            self._algo.after_test_episode(self._current_step, i, episode)
            self._experiment.save_test_env(self._current_step, i, self._test_env.summary())
            episodes.append(episode)
        # Log test metrics
        metrics = Episode.agregate_metrics(episodes)
        self._logger.log_print("Test", metrics, self._current_step)
        if metrics.score > self._best_score:
            self._best_score = metrics.score
            self._algo.save(f"{self._checkpoint}-{self._current_step}")
        self._algo.after_tests(episodes, self._current_step)

    def seed(self, seed_value: int):
        self._seed = seed_value
        import torch
        import random
        import os
        import numpy as np
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)
        self._env.seed(seed_value)
        self._test_env.seed(seed_value)

    def summary(self) -> dict:
        return {
                "env": self._env.summary(),
                "algorithm": self._algo.summary(),
                "seed": self._seed
            }

    @property
    def logdir(self) -> str:
        return self._logger.logdir