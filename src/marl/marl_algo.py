from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from rlenv.models import Episode, Transition, EpisodeBuilder, Observation, RLEnv
from . import logging


class RLAlgorithm(ABC):
    def __init__(self, env: RLEnv) -> None:
        super().__init__()
        self.env = env
        self.test_env = deepcopy(env)
        self.logger = logging.default()
        self._best_score = -float("inf")
        import os
        self._checkpoint = os.path.join(self.logger.logdir, "checkpoint")

    @abstractmethod
    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        """Get the action to perform given the input observation"""

    def save(self, to_path: str):
        """Save the algorithm state to the specified file."""
        raise NotImplementedError()

    def load(self, from_path: str):
        """Load the algorithm state from the specified file."""
        raise NotImplementedError()

    def before_tests(self):
        """Hook before tests, for instance to swap from training to testing policy."""

    def after_tests(self, time_step: int, episodes: list[Episode]):
        """
        Hook after tests. Logs test metrics by default and saves the model if required.
        Subclasses should swap from testing back to training policy too.
        """
        metrics = Episode.agregate_metrics(episodes)
        self.logger.log_print("Test", metrics, time_step)
        if metrics.score > self._best_score:
            self._best_score = metrics.score
            self.save(self.logger.logdir)

    def after_step(self, transition: Transition, step_num: int):
        """Hook after every training step."""

    def before_episode(self, episode_num: int):
        """Hook before every training and testing episode."""

    def after_episode(self, episode_num: int, episode: Episode):
        """Hook after every training episode."""

    def train(self, test_interval: int=200, n_tests: int=10, n_episodes: int=None, n_steps: int=None, quiet=False):
        """Start the training loop"""
        if not ((n_episodes is None) != (n_steps is None)):
            raise ValueError(f"Exactly one of n_episodes ({n_episodes}) and n_steps ({n_steps}) must be set !")
        if n_episodes is not None:
            self._train_episodes(n_episodes, test_interval, n_tests, quiet)
        else:
            self._train_steps(n_steps, test_interval, n_tests, quiet)

    def _train_steps(self, n_steps: int, test_interval: int, n_tests: int, quiet=False):
        """Train an agent and log on the basis of step numbers"""
        e = 0
        episode = EpisodeBuilder()
        obs = self.env.reset()
        for step in range(0, n_steps, test_interval):
            self.test(step, n_tests)
            stop = min(n_steps, step + test_interval)
            for i in tqdm(range(step, stop), leave=True, desc=f"Train {step}/{n_steps}", dynamic_ncols=True, disable=quiet):
                if episode.is_done:
                    episode = episode.build()
                    self.after_episode(e, episode)
                    self.logger.log("Train", episode.metrics, i)
                    e += 1
                    self.before_episode(e)
                    episode = EpisodeBuilder()
                    obs = self.env.reset()
                action = self.choose_action(obs)
                obs_, reward, done, info = self.env.step(action)
                transition = Transition(obs, action, reward, done, info, obs_)
                self.after_step(i, transition)
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
                self.before_episode(e)
                episode = EpisodeBuilder()
                obs = self.env.reset()
                while not episode.is_done:
                    action = self.choose_action(obs)
                    obs_, reward, done, info = self.env.step(action)
                    transition = Transition(obs, action, reward, done, info, obs_)
                    self.after_step(step, transition)
                    episode.add(transition)
                    obs = obs_
                    step += 1
                episode = episode.build()
                self.after_episode(e, episode)
                self.logger.log("Train", episode.metrics, e)
        self.test(e, n_tests)


    def test(self, time_step: int, ntests: int, quiet=False):
        """Test the agent"""
        self.before_tests()
        episodes: list[Episode] = []
        for i in tqdm(range(ntests), desc="Testing", unit="Episode", leave=True, disable=quiet):
            self.before_episode(i)
            episode = EpisodeBuilder()
            obs = self.test_env.reset()
            while not episode.is_done:
                action = self.choose_action(obs)
                new_obs, reward, done, info = self.test_env.step(action)
                transition = Transition(obs, action, reward, done, info, new_obs)
                episode.add(transition)
                obs = new_obs
            episodes.append(episode.build())
        self.after_tests(time_step, episodes)
