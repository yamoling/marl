import os
import shutil
import json
from rlenv import Episode, Observation
from marl.qlearning import QLearning
from marl.utils.algo_wrapper import AlgorithmWrapper
from marl.utils import alpha_num_order


class FileWrapper(AlgorithmWrapper):
    def __init__(self, algo: QLearning, directory: str):
        super().__init__(algo)
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.algo: QLearning = self.algo
        self.training = True
        self.training_qvalues: list = []
        self.testing_qvalues: list[list] = []

    def choose_action(self, observation: Observation):
        qvalues = self.algo.compute_qvalues(observation)
        if self.training:
            self.training_qvalues.append(qvalues.tolist())
        else:
            self.testing_qvalues[-1].append(qvalues.tolist())
        return super().choose_action(observation)

    def after_episode(self, episode_num: int, episode: Episode):
        episode_path = os.path.join(self.directory, "train", f"episode-{episode_num}.json")
        os.makedirs(os.path.dirname(episode_path), exist_ok=True)
        with open(episode_path, "w", encoding="utf-8") as f:
            json_data = episode.to_json()
            json_data["qvalues"] = self.training_qvalues
            json.dump(json_data, f)
        self.training_qvalues = []
        return super().after_episode(episode_num, episode)

    def before_episode(self, episode_num: int):
        if not self.training:
            self.testing_qvalues.append([])
        return super().before_episode(episode_num)

    def before_tests(self):
        self.training = False
        self.testing_qvalues = []
        return super().before_tests()

    def after_tests(self, episodes: list[Episode], time_step: int):
        self.training = True
        folder_path = os.path.join(self.directory, "test", f"step-{time_step}")
        self.save(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        # Log metrics
        metrics = Episode.agregate_metrics(episodes)
        with open(os.path.join(folder_path, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics.to_json(), f)
        # Move test videos to the appropriate folder
        video_folder = os.path.join(self.directory, "videos")
        video_paths = sorted(os.listdir(video_folder), key=alpha_num_order)
        for i, v in enumerate(video_paths):
            src = os.path.join(video_folder, v)
            dst = os.path.join(folder_path, f"{i}.mp4")
            shutil.move(src, dst)
        for i, (episode, qvalues) in enumerate(zip(episodes, self.testing_qvalues)):
            episode_path =  os.path.join(folder_path, f"{i}.json")
            with open(episode_path, "w", encoding="utf-8") as f:
                json_data = episode.to_json()
                json_data["qvalues"] = qvalues
                json.dump(json_data, f)
        return super().after_tests(episodes, time_step)
