import json
import os
from copy import deepcopy
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition
from tqdm import tqdm
from . import logging
from .marl_algo import RLAlgo
from .utils import defaults_to


class Runner:
    def __init__(
        self,
        env: RLEnv,
        algo: RLAlgo,
        logdir: str=None,
        test_env: RLEnv=None
    ):
        self._env = env
        self._test_env = defaults_to(test_env, deepcopy(env))
        self._algo = algo
        self._logger = logging.default(logdir)
        self._seed = None
        self._best_score = -float("inf")
        self._checkpoint = os.path.join(self._logger.logdir, "checkpoint")

    def train(self, test_interval: int=200, n_tests: int=10, n_episodes: int=None, n_steps: int=None, quiet=False) -> str:
        """Start the training loop"""
        with open(f"{self._logger.logdir}/experiment.json", "w", encoding="utf-8") as f:
            json.dump({
                "env": self._env.summary(),
                "training": {
                    "n_steps": n_steps,
                    "n_episodes": n_episodes,
                    "test_interval": test_interval,
                    "n_tests": n_tests
                },
                "algorithm": self._algo.summary()
            }, f, indent=4)

        if not ((n_episodes is None) != (n_steps is None)):
            raise ValueError(f"Exactly one of n_episodes ({n_episodes}) and n_steps ({n_steps}) must be set !")
        if n_episodes is not None:
            self._train_episodes(n_episodes, test_interval, n_tests, quiet)
        else:
            self._train_steps(n_steps, test_interval, n_tests, quiet)
        return self._logger.logdir

    def _train_steps(self, n_steps: int, test_interval: int, n_tests: int, quiet=False):
        """Train an agent and log on the basis of step numbers"""
        e = 0
        episode = EpisodeBuilder()
        obs = self._env.reset()
        self._algo.before_episode(0)
        for step in range(0, n_steps, test_interval):
            self.test(step, n_tests)
            stop = min(n_steps, step + test_interval)
            for i in tqdm(range(step, stop), leave=True, desc=f"Train {step}/{n_steps}", dynamic_ncols=True, disable=quiet):
                if episode.is_done:
                    episode = episode.build()
                    self._algo.after_episode(e, episode)
                    self._logger.log("Train", episode.metrics, i)
                    e += 1
                    self._algo.before_episode(e)
                    episode = EpisodeBuilder()
                    obs = self._env.reset()
                action = self._algo.choose_action(obs)
                obs_, reward, done, info = self._env.step(action)
                transition = Transition(obs, action, reward, done, info, obs_)
                self._algo.after_step(transition, i)
                episode.add(transition)
                obs = obs_
        self.test(n_steps, n_tests)

    def _train_episodes(self, n_episodes: int, test_interval: int, n_tests: int, quiet=False):
        """Train an agent and log on basis on episodes"""
        step = 0
        for e in range(0, n_episodes, test_interval):
            self.test(e, n_tests)
            stop = min(e + test_interval, n_episodes)
            for e in tqdm(range(e, stop), leave=True, unit="batch", desc=f"[Train {e}/{n_episodes}]", dynamic_ncols=True, disable=quiet):
                self._algo.before_episode(e)
                episode = EpisodeBuilder()
                obs = self._env.reset()
                while not episode.is_done:
                    action = self._algo.choose_action(obs)
                    obs_, reward, done, info = self._env.step(action)
                    transition = Transition(obs, action, reward, done, info, obs_)
                    self._algo.after_step(transition, step)
                    episode.add(transition)
                    obs = obs_
                    step += 1
                episode = episode.build()
                self._algo.after_episode(e, episode)
                self._logger.log("Train", episode.metrics, e)
        self.test(e, n_tests)


    def test(self, time_step: int, ntests: int, quiet=False):
        """Test the agent"""
        self._algo.before_tests()
        episodes: list[Episode] = []
        for i in tqdm(range(ntests), desc="Testing", unit="Episode", leave=True, disable=quiet):
            self._algo.before_episode(i)
            episode = EpisodeBuilder()
            obs = self._test_env.reset()
            while not episode.is_done:
                action = self._algo.choose_action(obs)
                new_obs, reward, done, info = self._test_env.step(action)
                transition = Transition(obs, action, reward, done, info, new_obs)
                episode.add(transition)
                obs = new_obs
            episodes.append(episode.build())
        # Log test metrics
        metrics = Episode.agregate_metrics(episodes)
        self._logger.log_print("Test", metrics, time_step)
        if metrics.score > self._best_score:
            self._best_score = metrics.score
            self._algo.save(f"{self._checkpoint}-{time_step}")
        self._algo.after_tests(episodes, time_step)

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