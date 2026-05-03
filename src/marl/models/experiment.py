import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence

import numpy as np
import numpy.typing as npt
import orjson
import torch
from marlenv.models import Space

from marl.utils import Serializable, stats
from marl.utils.stats import Dataset

from .run import Run

if TYPE_CHECKING:
    from marl.config import EnvConfig, TrainerConfig
    from marl.logging import LoggerType, TickColumn
    from marl.models.replay_episode import LightEpisodeSummary


@dataclass
class Experiment[A: Space, T: npt.ArrayLike](Serializable):
    env: EnvConfig[A]
    trainer: TrainerConfig[T]
    n_steps: int = 1_000_000
    logdir: str = "logs/test"
    test_env: EnvConfig[A] | None = None
    """Environment configuration to test the trained agent against. Defaults to `self.env`."""
    loggers: Collection[LoggerType] = field(default_factory=lambda: ["csv"])
    creation_timestamp: datetime | None = None

    def __post_init__(self):
        # Only create the timestamp the first time the experiment is created.
        # The other times, the attribute will already be set by the deserializer.
        if self.creation_timestamp is None:
            self.creation_timestamp = datetime.now()
        if not self.logdir.startswith("logs"):
            self.logdir = Path("logs", self.logdir).as_posix()

    def create_runs(self, seeds: int | Collection[int], n_tests: int, test_interval: int, save_weights: bool, save_actions: bool):
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        if self.test_env is None:
            self.test_env = self.env
        runs = [
            Run(
                seed,
                self.logpath / f"run-{seed}",
                self.trainer,
                self.env,
                self.test_env,
                self.n_steps,
                test_interval,
                n_tests,
                self.loggers,
                save_weights,
                save_actions,
            )
            for seed in seeds
        ]
        # Deepcopy to prevent modifying the original configs references
        return deepcopy(runs)

    @property
    def logpath(self):
        return Path(self.logdir)

    def save(self):
        self.to_file(self.logpath / "experiment.json")

    def run(
        self,
        seeds: int | Collection[int] = 1,
        gpu_strategy: Literal["scatter", "group"] = "group",
        save_weights: bool = True,
        save_actions: bool = True,
        n_tests: int = 1,
        test_interval: int = 5000,
        *,
        quiet: bool = False,
        device: Literal["cpu", "auto"] | int = "auto",
        render_tests: bool = False,
        n_jobs: int = torch.cuda.device_count(),
        disabled_gpus: Sequence[int] = (),
    ):
        """Train the Agent on the environment according to the experiment parameters."""
        from marl.runners import parallel_run, sequential_run

        if isinstance(seeds, int):
            seeds = list(range(seeds))
        runs = self.create_runs(seeds, n_tests, test_interval, save_weights, save_actions)
        if n_jobs <= 1 or len(runs) <= 1:
            return sequential_run(runs, device, gpu_strategy, quiet, render_tests, disabled_gpus)
        return parallel_run(runs, n_jobs, device, gpu_strategy, render_tests, disabled_gpus, quiet)

    def replay_episode(self, run_num: int, time_step: int, test_num: int, *, only_saved_actions: bool = False):
        """Replay the `test_num`th test episode at the `time_step`th test step from the `run_num`th run."""
        run = self.get_run(run_num)
        return run.replay_episode(time_step, test_num, only_saved_actions)

    def move(self, new_logdir: str):
        """Move an experiment to a new directory."""
        shutil.move(self.logdir, new_logdir)
        self.logdir = new_logdir
        self.save()

    @staticmethod
    def json_file(logdir: str):
        return os.path.join(logdir, "experiment.json")

    def get_run(self, run_num: int) -> Run[A, T]:
        rundir = self.rundirs[run_num]
        return Run.load(rundir, self.logger)

    @property
    def runs(self):
        """All the runs related to the experiment."""
        for rundir in self.rundirs:
            yield Run.from_file(rundir / "run.json")

    @property
    def rundirs(self):
        ls = sorted([f for f in os.listdir(self.logdir) if f.startswith("run_")])
        return [os.path.join(self.logdir, run) for run in ls]

    @staticmethod
    def is_experiment_directory(logdir: str) -> bool:
        """Check if a directory is an experiment directory."""
        try:
            return os.path.exists(os.path.join(logdir, "experiment.json"))
        except FileNotFoundError:
            return False

    @classmethod
    def find_experiment_directory(cls, subdir: str) -> str | None:
        """Find the experiment directory containing a given subdirectory."""
        if cls.is_experiment_directory(subdir):
            return subdir
        parent = os.path.dirname(subdir)
        if parent == subdir:
            return None
        return cls.find_experiment_directory(parent)

    @property
    def is_running(self):
        """Check if an experiment is running."""
        for run in self.runs:
            if run.is_running:
                return True
        return False

    def kill_runs(self):
        """Kill all runs of an experiment."""
        for run in self.runs:
            run.kill()

    def delete(self):
        shutil.rmtree(self.logdir)

    def get_tests_at(self, time_step: int):
        summary = list[LightEpisodeSummary]()
        for run in self.runs:
            summary += run.get_test_episodes(time_step)
        return summary

    def n_active_runs(self):
        return len([run for run in self.runs if run.is_running])

    def get_results_datasets(self, granularity: int, aggregate_by: TickColumn = "time_step"):
        results = self.get_results(granularity, aggregate_by)
        datasets = list[Dataset]()
        for category, stats_df in results.items():
            stats_df = stats_df.collect()
            columns = [col[5:] for col in stats_df.columns if col.startswith("mean-")]
            ticks = stats_df["ticks"].to_list()
            datasets += [
                Dataset(
                    logdir=self.logdir,
                    ticks=ticks,
                    label=col,
                    category=category,
                    mean=stats_df[f"mean-{col}"].to_numpy().astype(np.float32),
                    std=stats_df[f"std-{col}"].to_numpy().astype(np.float32),
                    min=stats_df[f"min-{col}"].to_numpy().astype(np.float32),
                    max=stats_df[f"max-{col}"].to_numpy().astype(np.float32),
                    ci95=stats_df[f"ci95-{col}"].to_numpy().astype(np.float32),
                )
                for col in columns
            ]
        return datasets

    def get_results(self, granularity: int, aggregate_by: TickColumn = "time_step"):
        """
        Return the category-wise metrics aggregated by rounded step buckets, or elapsed-time buckets when wall-time mode is enabled.

        E.g.: if the time steps are [1, 2, 3, 4, 5] and the granularity is 2, the time steps will be rounded to [0, 2, 2, 4, 4], and the metrics will be averaged for each time step, resulting in a dataframe with time steps [0, 2, 4].
        """
        runs = list(self.runs)
        # if self.env.is_multi_objective:
        #     qvalues = stats.compute_qvalues([run.qvalues_data(self.test_interval) for run in runs], self.logdir, replace_inf, self.qvalue_infos)
        return {
            "Test": stats.compute_experiment_results(
                [run.test_metrics for run in runs],
                aggregate_by,
                granularity,
            ),
            "Train": stats.compute_experiment_results(
                [run.train_metrics for run in runs],
                aggregate_by,
                granularity,
            ),
            "Training data": stats.compute_experiment_results(
                [run.training_data for run in runs],
                aggregate_by,
                granularity,
            ),
        }

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
    def get_parameters(logdir: str) -> dict[str, Any]:
        with open(Experiment.json_file(logdir), "rb") as f:
            return orjson.loads(f.read())

    def get_run_with_seed(self, seed: int):
        for run in self.runs:
            if run.seed == seed:
                return run
        return None
