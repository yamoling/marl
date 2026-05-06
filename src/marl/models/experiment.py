import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from signal import SIGINT
import logging
from typing import TYPE_CHECKING, Collection, Literal, Sequence, overload

import numpy as np
import numpy.typing as npt
import torch
from marlenv import MARLEnv

from marl.models.trainer import Trainer
from marl.utils import Serializable, stats
from marl.utils.stats import Dataset

from .run import Run

if TYPE_CHECKING:
    from marl.env import EnvConfig
    from marl.logging import LoggerType, TickColumn
    from marl.models import Trainer
    from marl.models.replay_episode import LightEpisodeSummary


EXPERIMENT_FILENAME = "experiment.json"


@dataclass
class Experiment[E: MARLEnv, T: Trainer](Serializable):
    env: EnvConfig[E]
    trainer: T
    n_steps: int = 1_000_000
    logdir: str = "logs/test"
    test_env: EnvConfig[E] | None = None
    """Environment configuration to test the trained agent against. Defaults to `self.env`."""
    loggers: Collection[LoggerType] = field(default_factory=lambda: ["csv"])
    creation_timestamp: datetime | None = None

    def __post_init__(self):
        if not self.logdir.startswith("logs"):
            self.logdir = Path("logs", self.logdir).as_posix()
        # Only create the timestamp the first time the experiment is created.
        # The other times, the attribute will already be set by the deserializer.
        is_new = self.creation_timestamp is None
        if is_new:
            self.creation_timestamp = datetime.now()
            if self.logpath.parts[-1].lower() in ("debug", "test", "tests"):
                logging.info(f"Discarding pre-existing experiment {self.logpath}.")
                self.delete()
            self.save()

    def create_runs(self, seeds: int | Collection[int], n_tests: int, test_interval: int, save_weights: bool, save_actions: bool):
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        if self.test_env is None:
            self.test_env = self.env
        runs = [
            Run(
                seed,
                (self.logpath / f"run-{seed}").as_posix(),
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

    @property
    def experiment_file(self):
        return self.logpath / EXPERIMENT_FILENAME

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

    def replay_episode(self, run_seed: int, time_step: int, test_num: int, *, only_saved_actions: bool = False):
        """Replay the `test_num`th test episode at the `time_step`th test step from the `run_num`th run."""
        run = self.get_run(run_seed)
        assert run is not None
        return run.replay_episode(time_step, test_num, only_saved_actions)

    def move(self, new_logdir: Path):
        """Move an experiment to a new directory."""
        shutil.move(self.logdir, new_logdir)
        self.logdir = new_logdir.as_posix()
        self.save()
        for run in self.runs:
            run.rundir = (new_logdir / run.runpath.parts[-1]).as_posix()
            run.save()

    @staticmethod
    def json_file(logdir: str | Path):
        logdir = Path(logdir)
        return logdir / EXPERIMENT_FILENAME

    @overload
    def get_run(self, run_seed: int, /): ...
    @overload
    def get_run(self, rundir: str, /): ...

    def get_run(self, seed_or_rundir: str | int):
        seed = None
        rundir = None
        match seed_or_rundir:
            case int(seed):
                pass
            case str(rundir):
                pass
            case other:
                raise ValueError(f"Invalid seed or rundir: {other}")
        for run in self.runs:
            if run.seed == seed or run.rundir == rundir:
                return run

    @property
    def runs(self):
        """All the runs related to the experiment."""
        for f in os.listdir(self.logdir):
            rundir = self.logpath / f
            if not rundir.is_dir():
                continue
            try:
                yield Run[E, npt.ArrayLike].load(self.logpath / f)
            except FileNotFoundError:
                pass

    @staticmethod
    def is_experiment_directory(logdir: str | Path) -> bool:
        """Check if a directory is an experiment directory."""
        logdir = Path(logdir)
        return Experiment.json_file(logdir).exists()

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
        return any(r.is_running for r in self.runs)

    def kill_runs(self):
        """Kill all runs of an experiment."""
        ppids = set[int]()
        n_killed = 0
        for run in self.runs:
            ppid = run.ppid
            if ppid is not None:
                ppids.add(ppid)
            if run.kill():
                n_killed += 1
        # If there was one single parent, we assume it was a parallel_runner and kill it as well
        if n_killed > 1 and len(ppids) == 1:
            ppid = ppids.pop()
            try:
                os.kill(ppid, SIGINT)
            except ProcessLookupError:
                pass

    @classmethod
    def load(cls, logdir: Path | str):
        json_file = cls.json_file(logdir)
        return cls.from_file(json_file)

    def save(self):
        self.to_file(self.experiment_file)

    def delete(self):
        print(f"Removing  experiment at {self.logpath}")
        shutil.rmtree(self.logpath)

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

    def copy(self, new_logdir: Path, copy_runs: bool = True):
        new_exp = deepcopy(self)
        new_exp.logdir = new_logdir.as_posix()
        new_exp.save()
        if not copy_runs:
            return new_exp
        for run in self.runs:
            new_rundir = new_logdir / run.runpath.parts[-1]
            run.runpath.replace(new_rundir)
            # shutil.copytree(run.rundir, new_rundir)
        return new_exp

    def get_run_with_seed(self, seed: int):
        for run in self.runs:
            if run.seed == seed:
                return run
        return None
