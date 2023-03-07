import json
import os
from copy import deepcopy
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition, Observation
from tqdm import tqdm
from . import logging
from .marl_algo import RLAlgo
from .utils import defaults_to


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
        self._episode_builder = EpisodeBuilder()
        self._episode_num = 0
        self._obs: Observation|None = None
        self.write_experiment_summary()

    def write_experiment_summary(self, train_summary: dict=None):
        with open(f"{self.logdir}/experiment.json", "w", encoding="utf-8") as f:
            json.dump({
                **self.summary(),
                "training": train_summary,
            }, f, indent=4)

    def train(self, n_steps: int, test_interval: int=200, n_tests: int=10, quiet=False) -> str:
        """Start the training loop"""
        self.write_experiment_summary({"n_steps": n_steps, "test_interval": test_interval, "n_tests": n_tests})
        #self._train_steps(n_steps, test_interval, n_tests, quiet)
        self.train_steps(n_steps, test_interval, n_tests, quiet)
        

    def train_steps(self, n_steps: int, test_interval: int, n_tests: int, quiet=False) -> str:
        """Start the training loop"""
        stop = self._current_step + n_steps
        if self._current_step == 0:
            self._obs = self._env.reset()
            self._algo.before_train_episode(self._episode_num)
        # TODO: tqdm
        while self._current_step < stop:
            if self._current_step % test_interval == 0:
                self.test(n_tests, quiet=quiet)
            if self._episode_builder.is_done:
                episode = self._episode_builder.build()
                self._algo.after_train_episode(self._episode_num, episode)
                self._logger.log("Train", episode.metrics, self._current_step)
                self._episode_num += 1
                self._algo.before_train_episode(self._episode_num)
                self._episode_builder = EpisodeBuilder()
                self._obs = self._env.reset()
            action = self._algo.choose_action(self._obs)
            obs_, reward, done, info = self._env.step(action)
            transition = Transition(self._obs, action, reward, done, info, obs_)
            self._algo.after_step(transition, self._current_step)
            self._episode_builder.add(transition)
            self._obs = obs_
            self._current_step += 1


    def _train_steps(self, n_steps: int, test_interval: int|None, n_tests: int, quiet=False):
        """Train an agent and log on the basis of step numbers"""
        e = 0
        episode = EpisodeBuilder()
        obs = self._env.reset()
        if test_interval is None:
            test_interval = n_steps
        self._algo.before_train_episode(e)
        for step in range(self._current_step, self._current_step + n_steps, test_interval):
            self.test(n_tests, quiet=quiet)
            stop = min(n_steps, step + test_interval)
            for i in tqdm(range(step, stop), leave=True, desc=f"Train {step}/{n_steps}", dynamic_ncols=True, disable=quiet):
                if episode.is_done:
                    episode = episode.build()
                    self._algo.after_train_episode(e, episode)
                    self._logger.log("Train", episode.metrics, i)
                    e += 1
                    self._algo.before_train_episode(e)
                    episode = EpisodeBuilder()
                    obs = self._env.reset()
                action = self._algo.choose_action(obs)
                obs_, reward, done, info = self._env.step(action)
                transition = Transition(obs, action, reward, done, info, obs_)
                self._algo.after_step(transition, i)
                episode.add(transition)
                obs = obs_
        self.test(n_steps, n_tests, quiet=quiet)

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