import os
import json
from rlenv.models import Episode, Metrics
from .deep_qwrapper import DeepQWrapper, IDeepQLearning


class ReplayWrapper(DeepQWrapper):
    def __init__(self, wrapped: IDeepQLearning, logdir: str) -> None:
        super().__init__(wrapped)
        self.logdir = logdir

    def test_directory(self, time_step: int, test_num: int = None) -> str:
        directory = os.path.join(self.logdir, "test", f"{time_step}")
        if test_num is not None:
            directory = os.path.join(directory, f"{test_num}")
        return directory
    
    def train_directory(self, episode_num: int) -> str:
        return os.path.join(self.logdir, "train", f"{episode_num}")
    
    def after_tests(self, episodes: list[Episode], time_step: int):
        # Log the actions and the metrics of individual episodes
        for test_num, episode in enumerate(episodes):
            directory = self.test_directory(time_step, test_num)
            self._log_episode(directory, episode.actions.tolist(), episode.metrics)
        # Save the model and log the agregated metrics
        test_dir = self.test_directory(time_step)
        self.save(test_dir)
        metrics = Episode.agregate_metrics(episodes)
        with open(os.path.join(test_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        return super().after_tests(episodes, time_step)

    @staticmethod
    def _log_episode(directory: str, actions: list, metrics: Metrics):
        os.makedirs(directory, exist_ok=True)
        with (open(os.path.join(directory, "actions.json"), "w") as a,
              open(os.path.join(directory, "metrics.json"), "w") as m):
            json.dump(actions, a)
            json.dump(metrics, m)
