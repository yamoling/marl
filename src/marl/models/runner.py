import os
import json
import torch
import pickle
from copy import deepcopy
from typing import Optional, Literal
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition
from tqdm import tqdm
from marl.utils import defaults_to

from .trainer import Trainer
from .algo import RLAlgo
from .run import Run


class Runner:
    def __init__(
        self,
        env: RLEnv,
        algo: RLAlgo,
        trainer: Trainer,
        run: Run,
        test_interval: int,
        n_steps: int,
        test_env: Optional[RLEnv] = None,
    ):
        self._trainer = trainer
        self._env = env
        self._test_env = defaults_to(test_env, lambda: deepcopy(env))
        self._algo = algo
        self._run = run
        self._test_interval = test_interval
        self._max_step = n_steps

    def _train_episode(self, step_num: int, episode_num: int, n_tests: int):
        episode = EpisodeBuilder()
        obs = self._env.reset()
        self._algo.new_episode()
        initial_value = self._algo.value(obs)
        while not episode.is_finished and step_num < self._max_step:
            step_num += 1
            if self._test_interval != 0 and step_num % self._test_interval == 0:
                self.test(n_tests, step_num)
            action, value, probs = self._algo.choose_action_extra(obs)
            obs_, reward, done, truncated, info = self._env.step(action)
            if step_num == self._max_step:
                truncated = True
            transition = Transition(obs, action, reward, done, info, obs_, truncated, value, probs)
            training_metrics = self._trainer.update_step(transition, step_num) | {"time_step": step_num}
            self._run.log_train_step(training_metrics)
            episode.add(transition)
            obs = obs_
        episode = episode.build({"initial_value": initial_value, "time_step": step_num})
        training_logs = self._trainer.update_episode(episode, episode_num, step_num) | {"time_step": step_num}
        self._run.log_train_episode(episode, training_logs)
        return episode

    def train(self, n_tests: int):
        """Start the training loop"""
        with open(os.path.join(self._run.rundir, "pid"), "w") as f:
            f.write(str(os.getpid()))
        episode_num = 0
        step = 0
        pbar = tqdm(total=self._max_step, desc="Training", unit="Step", leave=True)
        self.test(n_tests, 0)
        while step < self._max_step:
            episode = self._train_episode(step, episode_num, n_tests)
            episode_num += 1
            step += len(episode)
            pbar.update(len(episode))
        pbar.close()

    def test(self, ntests: int, time_step: int):
        """Test the agent"""
        self._algo.set_testing()
        test_dir = self._run.test_dir(time_step)
        self._algo.save(test_dir)
        episodes = list[Episode]()
        for _ in tqdm(range(ntests), desc="Testing", unit="Episode", leave=True):
            episode = EpisodeBuilder()
            obs = self._test_env.reset()
            self._algo.new_episode()
            intial_value = self._algo.value(obs)
            while not episode.is_finished:
                action, value, probs = self._algo.choose_action_extra(obs)
                new_obs, reward, done, truncated, info = self._test_env.step(action)
                transition = Transition(obs, action, reward, done, info, new_obs, truncated, value, probs)
                episode.add(transition)
                obs = new_obs
            episode = episode.build({"initial_value": intial_value, "time_step": time_step})
            episodes.append(episode)
        self._run.log_tests(episodes, time_step)
        self._algo.set_training()

    def _save_test_episode(self, directory: str, episode: Episode):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "env.pkl"), "wb") as e, open(os.path.join(directory, "actions.json"), "w") as a:
            try:
                pickle.dump(self._test_env, e)
            except (pickle.PicklingError, AttributeError):
                # AttributeError can be raised when the env is not pickleable
                pass
            json.dump(episode.actions.tolist(), a)

    def to(self, device: Literal["cpu", "auto", "cuda"] | torch.device):
        if isinstance(device, str):
            from marl.utils import get_device

            device = get_device(device)
        self._algo.to(device)
        self._trainer.to(device)
        return self

    def __del__(self):
        try:
            os.remove(os.path.join(self._run.rundir, "pid"))
        except FileNotFoundError:
            pass
