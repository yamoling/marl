import torch
from typing import Literal
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition
from tqdm import tqdm

import marl

from .trainer import Trainer
from .algo import RLAlgo
from .run import Run, RunHandle


class Runner:
    def __init__(self, env: RLEnv, algo: RLAlgo, trainer: Trainer, test_interval: int, n_steps: int, test_env: RLEnv):
        self._trainer = trainer
        self._env = env
        self._algo = algo
        self._test_interval = test_interval
        self._max_step = n_steps
        self._test_env = test_env

    def _train_episode(self, step_num: int, episode_num: int, n_tests: int, quiet: bool, run_handle: RunHandle):
        episode = EpisodeBuilder()
        obs = self._env.reset()
        self._algo.new_episode()
        initial_value = self._algo.value(obs)
        while not episode.is_finished and step_num < self._max_step:
            step_num += 1
            if self._test_interval != 0 and step_num % self._test_interval == 0:
                self.test(n_tests, step_num, quiet, run_handle)
            action = self._algo.choose_action(obs)
            obs_, reward, done, truncated, info = self._env.step(action)
            if step_num == self._max_step:
                truncated = True
            transition = Transition(obs, action, reward, done, info, obs_, truncated)
            training_metrics = self._trainer.update_step(transition, step_num)
            run_handle.log_train_step(training_metrics, step_num)
            episode.add(transition)
            obs = obs_
        episode = episode.build({"initial_value": initial_value})
        training_logs = self._trainer.update_episode(episode, episode_num, step_num)
        run_handle.log_train_episode(episode, step_num, training_logs)
        return episode

    def train(self, logdir: str, seed: int, n_tests: int, quiet: bool = False):
        """Start the training loop"""
        marl.seed(seed, self._env)
        self.randomize()
        episode_num = 0
        step = 0
        pbar = tqdm(total=self._max_step, desc="Training", unit="Step", leave=True, disable=quiet)
        with Run.create(logdir, seed) as run:
            self.test(n_tests, 0, quiet, run)
            while step < self._max_step:
                episode = self._train_episode(step, episode_num, n_tests, quiet, run)
                episode_num += 1
                step += len(episode)
                pbar.update(len(episode))
        pbar.close()

    def test(self, ntests: int, time_step: int, quiet: bool, run_handle: RunHandle):
        """Test the agent"""
        self._algo.set_testing()
        episodes = list[Episode]()
        for test_num in tqdm(range(ntests), desc="Testing", unit="Episode", leave=True, disable=quiet):
            self._test_env.seed(time_step + test_num)
            episode = EpisodeBuilder()
            obs = self._test_env.reset()
            intial_value = self._algo.value(obs)
            while not episode.is_finished:
                action = self._algo.choose_action(obs)
                new_obs, reward, done, truncated, info = self._test_env.step(action)
                transition = Transition(obs, action, reward, done, info, new_obs, truncated)
                episode.add(transition)
                obs = new_obs
            episode = episode.build({"initial_value": intial_value})
            episodes.append(episode)
        run_handle.log_tests(episodes, self._test_env, self._algo, time_step)
        self._algo.set_training()

    def randomize(self):
        self._algo.randomize()
        self._trainer.randomize()

    def to(self, device: Literal["cpu", "auto", "cuda"] | torch.device):
        if isinstance(device, str):
            from marl.utils import get_device

            device = get_device(device)
        self._algo.to(device)
        self._trainer.to(device)
        return self
