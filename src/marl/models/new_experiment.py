import os
import json
import time
from dataclasses import dataclass

from rlenv.models import RLEnv, Metrics, Episode, EpisodeBuilder, Transition
from rlenv import wrappers
import laser_env
from marl.models import RLAlgo, Runner
from marl.wrappers import ReplayWrapper
from marl.logging import Logger
from marl.utils import encode_b64_image, alpha_num_order, CorruptExperimentException


@dataclass
class ReplayEpisodeSummary:
    directory: str
    metrics: Metrics

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "directory": self.directory,
            "metrics": self.metrics.to_json(),
        }

@dataclass
class ReplayEpisode(ReplayEpisodeSummary):
    episode: Episode
    qvalues: list[list[list[float]]]
    frames: list[str]

    @property
    def name(self) -> str:
        return os.path.basename(self.directory)

    def to_json(self) -> dict:
        return {
            **super().to_json(),
            "episode": self.episode.to_json(),
            "qvalues": self.qvalues,
            "frames": self.frames,
        }



@dataclass
class Experiment:
    logdir: str
    summary: dict[str, ]
    _algo: RLAlgo | None = None
    _env: RLEnv | None = None

    def __init__(self, logdir: str, summary: dict[str, ]=None, algo: RLAlgo=None, env: RLEnv=None, save=True):
        """Create a new experiment. Either summary or algo and env must be provided."""
        if summary is None:
            assert algo is not None and env is not None
            summary = {
                "algorithm": algo.summary(),
                "env": env.summary(),
                "logdir": logdir,
                "timestamp_ms": int(time.time())
            }
        self.logdir = logdir
        self._algo = algo
        self._env = env
        self.summary = summary
        if save:
            self.save()

    @staticmethod
    def load(logdir: str) -> "Experiment":
        with open(os.path.join(logdir, "experiment.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        return Experiment(logdir, summary, save=False)
    
    def save(self):
        os.makedirs(self.logdir, exist_ok=True)
        with open(os.path.join(self.logdir, "experiment.json"), "w") as f:
            json.dump(self.summary, f)

    def create_runner(self, checkpoint: str=None, logger: Logger=None) -> Runner:
        start_step = 0
        if checkpoint is not None:
            self.algo.load(checkpoint)
            start_step = int(os.path.basename(checkpoint))
        
        return Runner(
            env=self.env, 
            algo=ReplayWrapper(self.algo, self.logdir),
            logger=logger,
            start_step=start_step,
        )
    
    @property
    def train_dir(self) -> str:
        return os.path.join(self.logdir, "train")
    
    @property
    def test_dir(self) -> str:
        return os.path.join(self.logdir, "test")
    
    def train_summary(self) -> list[ReplayEpisode]:
        base_dir = self.train_dir
        try: train_dirs = sorted(os.listdir(base_dir), key=alpha_num_order)
        except FileNotFoundError: return []
        summary = []
        for directory in train_dirs:
            episode_dir = os.path.join(base_dir, directory)
            try:
                with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                    summary.append(ReplayEpisodeSummary(episode_dir, Metrics(**json.load(f))))
            except FileNotFoundError:
                # The episode is not yet finished
                pass
        return summary
    
    def test_summary(self) -> list[ReplayEpisode]:
        summary = []
        base_dir = self.test_dir
        try: test_dirs = sorted(os.listdir(base_dir), key=alpha_num_order)
        except FileNotFoundError: return []
        try:
            for directory in test_dirs:
                episode_dir = os.path.join(base_dir, directory)
                with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                    summary.append(ReplayEpisodeSummary(episode_dir, Metrics(**json.load(f))))
            return summary    
        except FileNotFoundError:
            raise CorruptExperimentException(f"Test directory {base_dir} is corrupted.")
    
    @staticmethod
    def get_test_episodes(test_directory: str) -> list[ReplayEpisodeSummary]:
        summary = []
        for directory in sorted(os.listdir(test_directory), key=alpha_num_order):
            episode_dir = os.path.join(test_directory, directory)
            # Only consider episode directories.
            if os.path.isdir(episode_dir):
                with open(os.path.join(episode_dir, "metrics.json"), "r") as f:
                    summary.append(ReplayEpisodeSummary(episode_dir, Metrics(**json.load(f))))
        return summary

    @staticmethod
    def replay_episode(episode_folder: str) -> ReplayEpisode:
        with (open(os.path.join(episode_folder, "qvalues.json"), "r") as q, 
              open(os.path.join(episode_folder, "metrics.json"), "r") as m,
              open(os.path.join(episode_folder, "actions.json"), "r") as a,
              open(os.path.join(episode_folder, "env.json"), "r") as e
        ):
            qvalues = json.load(q)
            metrics = json.load(m)
            actions = json.load(a)
            env_summary = json.load(e)
        env = restore_env(env_summary, force_static=True)
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


    ###  Lazy loaded properties ###
    @property
    def algo(self):
        if self._algo is None:
            from marl import qlearning
            self._algo = qlearning.from_summary(self.summary["algorithm"])
        return self._algo
    
    @property
    def env(self):
        if self._env is None:
            self._env = restore_env(self.summary["env"])
        return self._env

def restore_env(env_summary: dict[str, ], force_static=False) -> RLEnv:
    if force_static:
        env = laser_env.StaticLaserEnv.from_summary(env_summary)
    else:
        match env_summary["name"]:
            case laser_env.DynamicLaserEnv.__name__:
                env = laser_env.DynamicLaserEnv.from_summary(env_summary)
            case laser_env.StaticLaserEnv.__name__:
                env = laser_env.StaticLaserEnv.from_summary(env_summary)
            case other: raise NotImplementedError(f"Cannot restore env {other}")
    return wrappers.from_summary(env, env_summary)
