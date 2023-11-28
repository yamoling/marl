import os
import shutil
from typing import Optional
from marl.models import Experiment, Run, ReplayEpisode


class ServerState:
    def __init__(self, logdir="logs"):
        self.experiments: dict[str, Experiment] = {}
        self.logdir = logdir

    def list_experiments(self) -> list[dict]:
        experiments = []
        for directory in os.listdir(self.logdir):
            directory = os.path.join(self.logdir, directory)
            try:
                experiments.append(Experiment.get_parameters(directory))
            except FileNotFoundError:
                # Not an experiment directory, ignore
                pass
        return experiments

    def load_experiment(self, logdir: str) -> Experiment:
        experiment = Experiment.load(logdir)
        self.experiments[logdir] = experiment
        return experiment

    def unload_experiment(self, logdir: str) -> Experiment | None:
        return self.experiments.pop(logdir, None)

    def get_runner_port(self, rundir: str) -> Optional[int]:
        return None

    def replay_episode(self, episode_dir: str) -> ReplayEpisode:
        longest_match = ""
        matching_experiment = None
        for logdir, experiment in self.experiments.items():
            if episode_dir.startswith(logdir) and len(logdir) > len(longest_match):
                longest_match = logdir
                matching_experiment = experiment
        if matching_experiment is None:
            raise ValueError(f"Could not find experiment for episode {episode_dir}")
        return matching_experiment.replay_episode(episode_dir)
