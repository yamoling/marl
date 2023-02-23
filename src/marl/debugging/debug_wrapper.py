import os
import json
from rlenv import Episode, Observation
from marl.qlearning import DeepQLearning
from marl.utils.algo_wrapper import AlgorithmWrapper


class QLearningDebugger(AlgorithmWrapper):
    def __init__(self, algo: DeepQLearning, directory: str):
        super().__init__(algo)
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.algo: DeepQLearning = self.algo
        self.training = True
        self.training_qvalues: list = []

    def choose_action(self, observation: Observation):
        qvalues = self.algo.compute_qvalues(observation)
        if self.training:
            self.training_qvalues.append(qvalues.tolist())
        return super().choose_action(observation)

    def after_episode(self, episode_num: int, episode: Episode):
        folder_path = os.path.join(self.directory, "train", f"{episode_num}")
        os.makedirs(folder_path, exist_ok=True)
        # Save qvalues
        qvalues_path = os.path.join(folder_path, "qvalues.json")
        with open(qvalues_path, "w", encoding="utf-8") as f:
            json.dump(self.training_qvalues, f)
            self.training_qvalues = []
        # Save metrics
        with open(os.path.join(folder_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(episode.metrics.to_json(), f)
        return super().after_episode(episode_num, episode)

    def before_tests(self):
        self.training = False
        return super().before_tests()

    def after_tests(self, episodes: list[Episode], time_step: int):
        self.training = True
        # Save the model
        folder_path = os.path.join(self.directory, "test", f"{time_step}")
        os.makedirs(folder_path, exist_ok=True)
        self.save(folder_path)
        # Log metrics
        metrics = Episode.agregate_metrics(episodes)
        with open(os.path.join(folder_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics.to_json(), f)
        return super().after_tests(episodes, time_step)
