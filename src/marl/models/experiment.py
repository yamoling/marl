from dataclasses import dataclass
from rlenv.models import Episode, EpisodeBuilder, Transition, Metrics
from copy import deepcopy
import os
import json
from laser_env import LaserEnv
import rlenv
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
    env: rlenv.RLEnv

    def __init__(self, logdir: str) -> None:
        self.logdir = logdir
        with open(os.path.join(logdir, "experiment.json"), "r") as f:
            self.summary = json.load(f)
        env_name = self.summary["env"]["name"]
        builder = rlenv.Builder(LaserEnv(self.summary["env"]["map"]))
        if "AgentIdWrapper" in env_name:
            builder.agent_id()
        
        time_limit = self.summary["env"].get("time_limit", None)
        if time_limit is not None:
            builder.time_limit(time_limit)
        self.env = builder.build()
        
    @property
    def train_dir(self) -> str:
        return os.path.join(self.logdir, "train")
    
    @property
    def test_dir(self) -> str:
        return os.path.join(self.logdir, "test")
    
    def train_summary(self) -> list[ReplayEpisode]:
        summary = []
        base_dir = self.train_dir
        for directory in self.list_trainings():
            episode_dir = os.path.join(base_dir, directory)
            with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                summary.append(ReplayEpisode(episode_dir, Metrics(**json.load(f))))
        return summary
    
    def test_summary(self) -> list[ReplayEpisode]:
        summary = []
        base_dir = self.test_dir
        for directory in self.list_tests():
            episode_dir = os.path.join(base_dir, directory)
            with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                summary.append(ReplayEpisode(episode_dir, Metrics(**json.load(f))))
        return summary
    
    def list_trainings(self) -> list[str]:
        try: return sorted(os.listdir(self.train_dir), key=alpha_num_order)
        except FileNotFoundError: return []
    
    def list_tests(self) -> list[str]:
        try: return sorted(os.listdir(self.test_dir), key=alpha_num_order)
        except FileNotFoundError: return []

    def list_test_episodes(self, step_num: int) -> list[str]:
        return sorted(os.listdir(os.path.join(self.test_dir, f"{step_num}")), key=alpha_num_order)


    def replay_episode(self, episode_folder: str) -> ReplayEpisode:
        with (open(os.path.join(episode_folder, "qvalues.json"), "r") as q, 
              open(os.path.join(episode_folder, "metrics.json"), "r") as m,
              open(os.path.join(episode_folder, "actions.json"), "r") as a
        ):
            qvalues = json.load(q)
            metrics = json.load(m)
            actions = json.load(a)
        env = deepcopy(self.env)
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
