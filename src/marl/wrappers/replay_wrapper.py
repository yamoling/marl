import numpy as np
import os
import json
from rlenv.models import Observation, Episode, Metrics
from .deep_qwrapper import DeepQWrapper, IDeepQLearning


class ReplayWrapper(DeepQWrapper):
    def __init__(self, wrapped: IDeepQLearning, logdir: str) -> None:
        super().__init__(wrapped)
        self.logdir = logdir
        self._training = True
        self.train_actions = []
        self.test_actions = []
        self.train_qvalues = []
        self.test_qvalues = []

    def test_directory(self, time_step: int, test_num: int = None) -> str:
        directory = os.path.join(self.logdir, "test", f"{time_step}")
        if test_num is not None:
            directory = os.path.join(directory, f"{test_num}")
        return directory
    
    def train_directory(self, episode_num: int) -> str:
        return os.path.join(self.logdir, "train", f"{episode_num}")
    
    def before_tests(self, time_step: int):
        self._training = False
        return super().before_tests(time_step)
    
    def before_train_episode(self, episode_num: int):
        self.train_actions = []
        self.train_qvalues = []
        return super().before_train_episode(episode_num)

    def after_train_episode(self, episode_num: int, episode: Episode):
        train_dir = self.train_directory(episode_num)
        self._log_episode(train_dir, self.train_actions, self.train_qvalues, episode.metrics)
        return super().after_train_episode(episode_num, episode)
    
    def before_test_episode(self, time_step: int, test_num: int):
        self.test_actions = []
        self.test_qvalues = []
        return super().before_test_episode(time_step, test_num)
    
    def after_test_episode(self, time_step: int, test_num: int, episode: Episode):
        directory = self.test_directory(time_step, test_num)
        self._log_episode(directory, self.test_actions, self.test_qvalues, episode.metrics)
        return super().after_test_episode(time_step, test_num, episode)

    def after_tests(self, episodes: list[Episode], time_step: int):
        test_dir = self.test_directory(time_step)
        self.save(test_dir)
        metrics = Episode.agregate_metrics(episodes)
        with open(os.path.join(test_dir, "metrics.json"), "w") as f:
            json.dump(metrics.to_json(), f)
        self._training = True
        return super().after_tests(episodes, time_step)

    def choose_action(self, observation: Observation) -> np.ndarray[np.int64]:
        qvalues = self.algo.compute_qvalues(observation)
        actions = super().choose_action(observation)
        if self._training:
            self.train_qvalues.append(qvalues.tolist())
            self.train_actions.append(actions.tolist())
        else:
            self.test_actions.append(actions.tolist())
            self.test_qvalues.append(qvalues.tolist())
        return actions
    
    def _log_episode(self, directory: str, actions: list, qvalues: list, metrics: Metrics):
        os.makedirs(directory, exist_ok=True)
        with (open(os.path.join(directory, "actions.json"), "w") as a,
              open(os.path.join(directory, "metrics.json"), "w") as m,
              open(os.path.join(directory, "qvalues.json"), "w") as q):
            json.dump(actions, a)
            json.dump(metrics.to_json(), m)
            json.dump(qvalues, q)