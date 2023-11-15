import os
import json
import shutil
from marl.models import Experiment, ReplayEpisodeSummary, Run, ReplayEpisode

from .messages import RunConfig, TrainConfig


class ServerState:
    def __init__(self, logdir="logs"):
        self.experiments: dict[str, Experiment] = {}
        self.logdir = logdir

    def list_experiments(self):
        experiments = []
        for directory in os.listdir(self.logdir):
            directory = os.path.join(self.logdir, directory)
            try:
                experiments.append(Experiment.get_parameters(directory))
            except FileNotFoundError:
                # Not an experiment directory, ignore
                pass
        return experiments

    def get_experiment_parameters(self, logdir: str) -> dict:
        with open(os.path.join(logdir, "experiment.json"), "r") as f:
            return json.load(f)

    def unload_experiment(self, logdir: str) -> Experiment | None:
        return self.experiments.pop(logdir, None)

    def delete_experiment(self, logdir: str):
        try:
            experiment = self.unload_experiment(logdir)
            if experiment is None:
                experiment = Experiment.load(logdir)
            for run in experiment.runs:
                run.stop()
            shutil.rmtree(logdir)
        except FileNotFoundError:
            raise ValueError(f"Experiment {logdir} could not be deleted !")

    def create_runner(self, logdir: str, run_config: RunConfig):
        """Creates a runner for the given experiment and returns their loggers"""
        if logdir not in self.experiments:
            raise ValueError(f"Experiment {logdir} not found")
        experiment = self.experiments[logdir]
        raise NotImplementedError()

    def stop_runner(self, rundir: str):
        logdir = os.path.dirname(rundir)
        if logdir not in self.experiments:
            raise ValueError(f"Experiment {rundir} not found")
        self.experiments[logdir].stop_runner(rundir)

    def restart_runner(self, rundir: str, train_config: TrainConfig):
        logdir = os.path.dirname(rundir)
        if logdir not in self.experiments:
            self.experiments[logdir] = Experiment.load(logdir)
        raise NotImplementedError()

    def delete_runner(self, rundir: str):
        shutil.rmtree(rundir)

    def get_runner_port(self, rundir: str) -> int:
        run = Run.load(rundir)
        return run.get_port()

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
