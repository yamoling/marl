from dataclasses import dataclass
from rlenv.models import Episode, EpisodeBuilder, Transition, Metrics
from copy import deepcopy
import os
import json
import tempfile
from laser_env import LaserEnv
import laser_env
import rlenv
from marl import RLAlgo
from marl.utils.others import encode_b64_image, alpha_num_order


@dataclass
class ReplayEpisode:
    directory: str
    metrics: Metrics
    episode: Episode | None = None
    qvalues: list[list[list[float]]] | None = None
    frames: list[str] | None = None

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "directory": self.directory,
            "episode": None if self.episode is None else self.episode.to_json(),
            "metrics": self.metrics.to_json(),
            "qvalues": self.qvalues,
            "frames": self.frames,
        }

@dataclass
class Experiment:
    logdir: str
    summary: dict

    def __init__(self, logdir: str, summary: dict, save=True) -> None:
        self.logdir = logdir
        self.summary = summary
        if save:
            self.save()
        
    @staticmethod
    def load(logdir: str) -> "Experiment":
        with open(os.path.join(logdir, "experiment.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        return Experiment(logdir, summary, save=False)
    
    def save(self):
        with open(os.path.join(self.logdir, "experiment.json"), "w") as f:
            json.dump(self.summary, f)

    def save_train_env(self, train_num: int, env_summary: dict):
        directory = os.path.join(self.train_dir, f"{train_num}")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "env.json"), "w") as f:
            json.dump(env_summary, f)

    def save_test_env(self, time_step: int, test_num: int, env_summary: dict):
        directory = os.path.join(self.test_dir, f"{time_step}", f"{test_num}")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "env.json"), "w") as f:
            json.dump(env_summary, f)

    @property
    def train_dir(self) -> str:
        return os.path.join(self.logdir, "train")
    
    @property
    def test_dir(self) -> str:
        return os.path.join(self.logdir, "test")

    def from_checkpoint(self, checkpoint_dir: str) -> RLAlgo:
        algo_wrappers: str = self.summary["algorithm"]["name"]
        algo_wrappers = algo_wrappers.replace('(', ')').split(')')
        print(algo_wrappers)
        
    
    def train_summary(self) -> list[ReplayEpisode]:
        summary = []
        base_dir = self.train_dir
        for directory in self.list_trainings():
            episode_dir = os.path.join(base_dir, directory)
            try:
                with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                    summary.append(ReplayEpisode(episode_dir, Metrics(**json.load(f))))
            except FileNotFoundError:
                # The episode is not yet finished
                pass
        return summary
    
    def test_summary(self) -> list[ReplayEpisode]:
        summary = []
        base_dir = self.test_dir
        for directory in self.list_tests():
            episode_dir = os.path.join(base_dir, directory)
            with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                summary.append(ReplayEpisode(episode_dir, Metrics(**json.load(f))))
        return summary
    
    def test_episode_summary(self, test_directory: str) -> list[ReplayEpisode]:
        summary = []
        for directory in sorted(os.listdir(test_directory), key=alpha_num_order):
            episode_dir = os.path.join(test_directory, directory)
            # Only consider episode directories.
            if os.path.isdir(episode_dir):
                with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                    summary.append(ReplayEpisode(episode_dir, Metrics(**json.load(f))))
        return summary
    
    def list_trainings(self) -> list[str]:
        try: return sorted(os.listdir(self.train_dir), key=alpha_num_order)
        except FileNotFoundError: return []
    
    def list_tests(self) -> list[str]:
        try: return sorted(os.listdir(self.test_dir), key=alpha_num_order)
        except FileNotFoundError: return []

    def list_test_episodes(self, time_step: int) -> list[str]:
        return sorted(os.listdir(os.path.join(self.test_dir, f"{time_step}")), key=alpha_num_order)

    @staticmethod
    def restore_env(episode_folder: str) -> rlenv.RLEnv:
        with open(os.path.join(episode_folder, "env.json"), "r") as f:
            env_summary: dict[str, str] = json.load(f)
        # Observation type
        obs_dtype = None
        for enum in laser_env.ObservationType:
            if enum.name == env_summary["obs_type"]:
                obs_dtype = enum
                break
        if obs_dtype is None:
            raise ValueError(f"Observation type {env_summary['obs_type']} not found.")
        
        # File content
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(env_summary["map_file_content"])
        builder = rlenv.Builder(LaserEnv(f.name, enum))
        os.remove(f.name)
        
        # Wrappers
        wrappers: list[str] = env_summary["name"].replace('(', ')').split(')')
        if "AgentIdWrapper" in wrappers:
            builder.agent_id()
        if "TimeLimitWrapper" in wrappers:
            time_limit: int = env_summary["time_limit"]
            builder.time_limit(time_limit)
        return builder.build()

    def replay_episode(self, episode_folder: str) -> ReplayEpisode:
        with (open(os.path.join(episode_folder, "qvalues.json"), "r") as q, 
              open(os.path.join(episode_folder, "metrics.json"), "r") as m,
              open(os.path.join(episode_folder, "actions.json"), "r") as a
        ):
            qvalues = json.load(q)
            metrics = json.load(m)
            actions = json.load(a)
        env = self.restore_env(episode_folder)
        obs = env.reset()
        frames = [encode_b64_image(env.render('rgb_array'))]
        episode = EpisodeBuilder()
        for action in actions:
            obs_, reward, done, info = env.step(action)
            episode.add(Transition(obs, action, reward, done, info, obs_))
            frames.append(encode_b64_image(env.render('rgb_array')))
            obs = obs_
        return ReplayEpisode(
            directory=episode_folder, 
            episode=episode.build(), 
            qvalues=qvalues, 
            frames=frames, 
            metrics=Metrics(**metrics)
        )
