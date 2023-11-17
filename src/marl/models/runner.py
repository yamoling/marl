import os
import json
import torch
import pickle
from copy import deepcopy
from typing import Optional
from rlenv.models import RLEnv, Episode, EpisodeBuilder, Transition
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
        test_env: Optional[RLEnv] = None,
        quiet=False,
    ):
        self._trainer = trainer
        self._env = env
        self._test_env = defaults_to(test_env, lambda: deepcopy(env))
        self._algo = algo
        self._logger = logger
        self._test_interval = test_interval
        self._quiet = quiet
        self._max_step = n_steps

    def _train_episode(self, step_num: int, episode_num: int, n_tests: int) -> Optional[Episode]:
        episode = EpisodeBuilder()
        obs = self._env.reset()
        self._algo.new_episode()
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
        self._trainer.update_episode(episode, episode_num, step_num)
        return episode

    def train(self, n_tests: int):
        """Start the training loop"""
        with open(os.path.join(self.rundir, "pid"), "w") as f:
            f.write(str(os.getpid()))
        episode_num = 0
        step = 0
        pbar = tqdm(total=self._max_step, desc="Training", unit="Step", leave=True, disable=self._quiet)
        while step < self._max_step:
            episode = self._train_episode(step, episode_num, n_tests)
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
            self._algo.new_episode()
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
        agg = agregate_metrics([e.metrics for e in episodes], skip_keys={"timestamp_sec", "time_step"})
        self._logger.print("test", agg)
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

    def to(self, device: str | torch.device):
        if isinstance(device, str):
            from marl.utils import get_device

            device = get_device(device)  # type: ignore
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


def agregate_metrics(
    all_metrics: list[dict[str, float]],
    only_avg=False,
    skip_keys: Optional[set[str]] = None,
) -> dict[str, float]:
    """Aggregate a list of metrics into min, max, avg and std."""
    import numpy as np

    if skip_keys is None:
        skip_keys = set()
    all_values: dict[str, list[float]] = {}
    for metrics in all_metrics:
        for key, value in metrics.items():
            if key not in all_values:
                all_values[key] = []
            all_values[key].append(value)
    res = {}
    if only_avg:
        for key, values in all_values.items():
            res[key] = float(np.average(np.array(values)))
    else:
        for key, values in all_values.items():
            if key not in skip_keys:
                values = np.array(values)
                res[f"avg_{key}"] = float(np.average(values))
                res[f"std_{key}"] = float(np.std(values))
                res[f"min_{key}"] = float(values.min())
                res[f"max_{key}"] = float(values.max())
    return res
