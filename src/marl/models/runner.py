import os
import json
from copy import deepcopy
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition, Observation
from tqdm import tqdm
from marl.logging import Logger
from marl.utils import defaults_to

from .algo import RLAlgo


class Runner:
    def __init__(
        self,
        env: RLEnv,
        algo: RLAlgo,
        logger: Logger,
        n_steps: int,
        test_interval: int,
        test_env: RLEnv=None,
        start_step=0,
    ):
        self._env = env
        self._test_env = defaults_to(test_env, lambda: deepcopy(env))
        self._algo = algo
        self._logger = logger
        self._seed = None
        self._best_score = -float("inf")
        self._checkpoint = os.path.join(self.rundir, "checkpoint")
        self._current_step = start_step
        self._episode_builder = None
        self._episode_num = 0
        self._obs: Observation | None = None
        self._n_steps = n_steps
        self._test_interval = test_interval

    def _before_train_episode(self):
        self._episode_builder = EpisodeBuilder()
        self._obs = self._env.reset()
        self._algo.before_train_episode(self._episode_num)

    def train(self, n_tests: int, quiet=False) -> str:
        """Start the training loop"""
        with open(os.path.join(self.rundir, "pid"), "w") as f:
            f.write(str(os.getpid()))
        stop = self._current_step + self._n_steps
        if self._episode_num == 0:
            self._before_train_episode()
        for i in tqdm(range(self._current_step, stop), desc="Training", unit="Step", leave=True, disable=quiet):
            self._current_step = i
            if self._current_step % self._test_interval == 0:
                self.test(n_tests, quiet=quiet)
            if self._episode_builder.is_done:
                episode = self._episode_builder.build()
                self._algo.after_train_episode(self._episode_num, episode)
                self._logger.log("train", episode.metrics, self._current_step - len(episode))
                self._episode_num += 1
                self._before_train_episode()
            action = self._algo.choose_action(self._obs)
            obs_, reward, done, info = self._env.step(action)
            transition = Transition(self._obs, action, reward, done, info, obs_)
            self._algo.after_step(transition, self._current_step)
            self._episode_builder.add(transition)
            self._obs = obs_
            self._current_step += 1
        if self._current_step % self._test_interval == 0:
            self.test(n_tests, quiet=quiet)
        self._logger.close()

    def test(self, ntests: int, quiet=False):
        """Test the agent"""
        self._algo.before_tests(self._current_step)
        test_dir = os.path.join(self.rundir, "test", f"{self._current_step}")
        self._algo.save(test_dir)
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
            self._save_test_episode(os.path.join(test_dir, f"{i}"), episode)
            self._logger.log("test", episode.metrics, self._current_step)
            episodes.append(episode)
        self._algo.after_tests(episodes, self._current_step)

    def _save_test_episode(self, directory: str, episode: Episode):
        os.makedirs(directory, exist_ok=True)
        with (open(os.path.join(directory, "env.json"), "w") as e,
              open(os.path.join(directory, "actions.json"), "w") as a):
            json.dump(self._test_env.summary(), e)
            json.dump(episode.actions.tolist(), a)


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

    @property
    def rundir(self) -> str:
        return self._logger.logdir
    

    def __del__(self):
        self._logger.close()
        try:
            os.remove(os.path.join(self.rundir, "pid"))
        except FileNotFoundError:
            pass
