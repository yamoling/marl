import os
import shutil
from copy import deepcopy
from dataclasses import dataclass

import orjson

from marl.utils import default_serialization, stats
from marl.models.run import Run
from marl.models.replay_episode import LightEpisodeSummary


@dataclass
class LightExperiment:
    logdir: str
    test_interval: int
    n_steps: int
    creation_timestamp: int

    @staticmethod
    def get_parameters(logdir: str) -> dict:
        """Get the parameters of an experiment."""
        with open(os.path.join(logdir, "experiment.json"), "r") as f:
            return orjson.loads(f.read())

    def move(self, new_logdir: str):
        """Move an experiment to a new directory."""
        shutil.move(self.logdir, new_logdir)
        self.logdir = new_logdir
        self.save()

    def save(self):
        os.makedirs(self.logdir, exist_ok=True)

        with open(os.path.join(self.logdir, "experiment.json"), "wb") as f:
            f.write(orjson.dumps(self, default=default_serialization, option=orjson.OPT_SERIALIZE_NUMPY))

    @property
    def runs(self):
        for run in os.listdir(self.logdir):
            if run.startswith("run_"):
                try:
                    yield Run.load(os.path.join(self.logdir, run))
                except Exception:
                    pass

    @staticmethod
    def is_experiment_directory(logdir: str) -> bool:
        """Check if a directory is an experiment directory."""
        try:
            return os.path.exists(os.path.join(logdir, "experiment.json"))
        except FileNotFoundError:
            return False

    @staticmethod
    def find_experiment_directory(subdir: str) -> str | None:
        """Find the experiment directory containing a given subdirectory."""
        if LightExperiment.is_experiment_directory(subdir):
            return subdir
        parent = os.path.dirname(subdir)
        if parent == subdir:
            return None
        return LightExperiment.find_experiment_directory(parent)

    @property
    def is_running(self):
        """Check if an experiment is running."""
        for run in self.runs:
            if run.is_running:
                return True
        return False

    def delete(self):
        shutil.rmtree(self.logdir)

    def get_tests_at(self, time_step: int):
        summary = list[LightEpisodeSummary]()
        for run in self.runs:
            summary += run.get_test_episodes(time_step)
        return summary

    @property
    def train_dir(self):
        return os.path.join(self.logdir, "train")

    @property
    def test_dir(self):
        return os.path.join(self.logdir, "test")

    @property
    def qvalue_infos(self):
        param = LightExperiment.get_parameters(self.logdir)
        labels = param["env"]["reward_space"]["labels"]
        n_agents = param["env"]["n_agents"]
        return (labels, n_agents)

    def n_active_runs(self):
        return len([run for run in self.runs if run.is_running])

    def get_experiment_results(self, replace_inf=False):
        """Get all datasets of an experiment. If no qvalues were logged, the dataframe is empty"""
        runs = list(self.runs)
        datasets = stats.compute_datasets([run.test_metrics for run in runs], self.logdir, replace_inf, suffix=" [test]")
        datasets += stats.compute_datasets(
            [run.train_metrics(self.test_interval) for run in runs], self.logdir, replace_inf, suffix=" [train]"
        )
        datasets += stats.compute_datasets([run.training_data(self.test_interval) for run in runs], self.logdir, replace_inf)
        qvalues = stats.compute_qvalues([run.qvalues_data(self.test_interval) for run in runs], self.logdir, replace_inf, self.qvalue_infos)

        return datasets, qvalues 

    def copy(self, new_logdir: str, copy_runs: bool = True):
        new_exp = deepcopy(self)
        new_exp.logdir = new_logdir
        new_exp.save()
        if copy_runs:
            for run in self.runs:
                new_rundir = run.rundir.replace(self.logdir, new_logdir)
                shutil.copytree(run.rundir, new_rundir)
        return new_exp

    @staticmethod
    def load(logdir: str):
        with open(os.path.join(logdir, "experiment.json"), "r") as f:
            data = orjson.loads(f.read())
        return LightExperiment(logdir, data["test_interval"], data["n_steps"], data["creation_timestamp"])
