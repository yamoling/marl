import os
import json
import torch
import pickle
from copy import deepcopy
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition, Metrics
from tqdm import tqdm
from marl.logging import Logger
from marl.utils import defaults_to

from .trainer import Trainer
from .algo import RLAlgo


class Runner:
    def __init__(
        self,
        env: RLEnv,
        algo: RLAlgo,
        trainer: Trainer,
        logger: Logger,
        test_interval: int,
        n_steps: int,
        test_env: RLEnv=None,
        quiet=False
    ):
        self._trainer = trainer
        self._env = env
        self._test_env = defaults_to(test_env, lambda: deepcopy(env))
        self._algo = algo
        self._logger = logger
        self._test_interval = test_interval
        self._quiet = quiet
        self._max_step = n_steps


    def _train_episode(self, step_num: int, episode_num: int, n_tests: int) -> Episode:
        episode = EpisodeBuilder()
        obs = self._env.reset()
        initial_value = self._algo.value(obs)
        while not episode.is_finished and step_num < self._max_step:
            if self._test_interval != 0 and step_num % self._test_interval == 0:
                self.test(n_tests, step_num)
            action = self._algo.choose_action(obs)
            obs_, reward, done, truncated, info = self._env.step(action)
            transition = Transition(obs, action, reward, done, info, obs_, truncated)
            self._trainer.update_step(transition, step_num)
            episode.add(transition)
            obs = obs_
            step_num += 1
        if not episode.is_finished:
            return None
        episode = episode.build({"initial_value": initial_value})
        self._trainer.update_episode(episode, step_num)
        return episode

    def train(self, n_tests: int) -> str:
        """Start the training loop"""
        with open(os.path.join(self.rundir, "pid"), "w") as f:
            f.write(str(os.getpid()))
        episode_num = 0
        step = 0
        pbar = tqdm(total=self._max_step, desc="Training", unit="Step", leave=True, disable=self._quiet)
        while step < self._max_step:
            episode = self._train_episode(step, episode_num,n_tests)
            episode_num += 1
            if episode is None:
                episode_length = self._max_step - step
            else:
                episode_length = len(episode)
                self._logger.log("train", episode.metrics, step)
            step += episode_length
            pbar.update(episode_length)
        self.test(n_tests, self._max_step)
        self._logger.close()

    def test(self, ntests: int, time_step: int):
        """Test the agent"""
        self._algo.set_testing()
        test_dir = os.path.join(self.rundir, "test", f"{time_step}")
        self._algo.save(test_dir)
        episodes = list[Episode]()
        for i in tqdm(range(ntests), desc="Testing", unit="Episode", leave=True, disable=self._quiet):
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
            self._save_test_episode(os.path.join(test_dir, f"{i}"), episode)
            self._logger.log("test", episode.metrics, time_step)
            episodes.append(episode)
        agg = Metrics.agregate([e.metrics for e in episodes], skip_keys={"timestamp_sec", "time_step"})
        self._logger.print("test", agg)
        self._algo.set_training()

    def _save_test_episode(self, directory: str, episode: Episode):
        os.makedirs(directory, exist_ok=True)
        with (open(os.path.join(directory, "env.pkl"), "wb") as e,
              open(os.path.join(directory, "actions.json"), "w") as a):
            pickle.dump(self._test_env, e)
            json.dump(episode.actions.tolist(), a)

    def to(self, device: str|torch.device):
        if isinstance(device, str):
            from marl.utils import get_device
            device = get_device(device)
        self._algo.to(device)
        self._trainer.to(device)
        return self

    @property
    def rundir(self) -> str:
        return self._logger.logdir
    

    def __del__(self):
        self._logger.close()
        try:
            os.remove(os.path.join(self.rundir, "pid"))
        except FileNotFoundError:
            pass
