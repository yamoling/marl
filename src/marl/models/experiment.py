import os
import json
import time
import shutil
from dataclasses import dataclass
from copy import deepcopy
from typing import Literal

from rlenv.models import RLEnv, Metrics, EpisodeBuilder, Transition
from rlenv import wrappers
import laser_env
from marl import logging
from marl.wrappers import ReplayWrapper
from marl.utils import encode_b64_image, ExperimentAlreadyExistsException
from marl.qlearning import IQLearning

from .runner import Runner
from .algo import RLAlgo
from .replay_episode import ReplayEpisode, ReplayEpisodeSummary
from .run import Run


@dataclass
class Experiment:
    logdir: str
    _runs: list[Run]
    _summary: dict[str, ]
    _algo: RLAlgo | None = None
    _env: RLEnv | None = None

    def __init__(self, logdir: str, summary: dict[str, ]=None, algo: RLAlgo=None, env: RLEnv=None):
        """This constructor should not be called directly. Use Experiment.create() or Experiment.load() instead."""
        if summary is None:
            assert algo is not None and env is not None
            summary = {
                "algorithm": algo.summary(),
                "env": env.summary(),
                "logdir": logdir,
                "timestamp_ms": int(time.time())
            }
        # Update the logdir in the summary in case the experiment has been manually moved.
        summary["logdir"] = logdir
        self._runs = [Run(os.path.join(logdir, run)) for run in os.listdir(logdir) if run.startswith("run_")]
        # Update the runs in the summary
        summary["runs"] = [run.to_json() for run in self._runs]
        self.logdir = logdir
        self._algo = algo
        self._env = env
        self._summary = summary
        

    @staticmethod
    def create(logdir: str, algo: RLAlgo, env: RLEnv) -> "Experiment":
        """Create a new experiment."""
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        try:
            # Remove the test and debug logs
            if logdir in ["logs/test", "logs/debug", "logs/tests"]:
                shutil.rmtree(logdir)
        except FileNotFoundError: pass
        try:
            os.makedirs(logdir, exist_ok=False)
        except FileExistsError:
            raise ExperimentAlreadyExistsException(logdir)
        experiment =  Experiment(logdir, None, algo=algo, env=env)
        experiment.save()
        return experiment

    @staticmethod
    def load(logdir: str) -> "Experiment":
        """Load an existing experiment."""
        with open(os.path.join(logdir, "experiment.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        return Experiment(logdir, summary)
    
    def save(self):
        os.makedirs(self.logdir, exist_ok=True)
        with open(os.path.join(self.logdir, "experiment.json"), "w") as f:
            json.dump(self._summary, f)

    def create_runner(self, logger: Literal["web", "tensorboard", "both"]="both", checkpoint: str=None, seed: int=None, forced_rundir: str=None, quiet=True) -> Runner:
        if forced_rundir is not None:
            rundir = forced_rundir
        else:
            rundir = os.path.join(self.logdir, f"run_{time.time()}")
        algo = deepcopy(self.algo)
        env = deepcopy(self.env)
        if checkpoint is not None:
            algo.load(checkpoint)
            try:
                shutil.copytree(checkpoint, rundir)
            except FileExistsError:
                pass
        match logger:
            case "web": logger = logging.WSLogger(rundir, quiet)
            case "tensorboard": logger = logging.TensorBoardLogger(rundir, quiet)
            case "both": logger = logging.MultiLogger(
                    rundir,
                    logging.WSLogger(rundir),
                    logging.TensorBoardLogger(rundir),
                    quiet=quiet
                )
        if issubclass(algo.__class__, IQLearning):
            algo=ReplayWrapper(algo, rundir)
        runner = Runner(
            env=env,
            algo=algo,
            logger=logger
        )
        if seed is not None:
            runner.seed(seed)
        self._runs.append(Run(rundir))
        return runner

    def stop_runner(self, rundir: str):
        """Stops the runner at the given rundir."""
        for i, run in enumerate(self._runs):
            if run.rundir == rundir:
                run = self._runs.pop(i)
                run.stop()
                return
        # If the run was not found, raise an error
        raise ValueError("This rundir does not exist.")

    def restore_runner(self, rundir: str):
        """Retrieve the runner state and restart it if it is not running"""
        run = Run(rundir)
        if run.is_running:
            raise ValueError("This run is already running.")
        runner = self.create_runner(checkpoint=run.latest_checkpoint, logger="both", forced_rundir=rundir)
        runner._current_step = run.current_step
        return runner

    def delete_run(self, rundir: str):
        for i, run in enumerate(self._runs):
            if run.rundir == rundir:
                run = self._runs.pop(i)
                run.delete()
        # If the run was not found, raise an error        
        raise ValueError("This rundir does not exist.")
    
    @property
    def train_dir(self) -> str:
        return os.path.join(self.logdir, "train")
    
    @property
    def test_dir(self) -> str:
        return os.path.join(self.logdir, "test")
    
    def test_metrics(self) -> dict[str, Metrics]:
        metrics: dict[str, list[Metrics]] = {}
        for run in self._runs:
            for time_step, m in run.test_metrics().items():
                # Add the metric to the others of the same time step
                if time_step not in metrics:
                    metrics[time_step] = [m]
                else:
                    metrics[time_step].append(m)
        return {episode: Metrics.agregate(m, only_avg=True) for episode, m in metrics.items()}
    
    def get_test_episodes(self, time_step: int) -> list[ReplayEpisodeSummary]:
        summary = []
        for run in self._runs:
            summary += run.get_test_episodes(time_step)
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
            self._algo = qlearning.from_summary(self._summary["algorithm"])
        return self._algo
    
    @property
    def env(self):
        if self._env is None:
            self._env = restore_env(self._summary["env"])
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
