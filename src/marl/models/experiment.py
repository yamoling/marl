import logging
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from signal import SIGINT
from typing import Collection, Literal, overload

import numpy as np
import orjson
import polars as pl
import torch
from marlenv import MARLEnv
from polars import selectors as cs

from marl.env import EnvConfig
from marl.logging import LoggerType, TickColumn
from marl.models.trainer import Trainer
from marl.utils import DeviceLike, Serializable, stats

from .dataset import Dataset
from .run import LightRun, Run

EXPERIMENT_FILENAME = "experiment.json"


@dataclass
class LightExperiment[E: MARLEnv, T: Trainer](Serializable):
    """
    An Experiment essentially defined by an environment and a trainer. Use `Experiment.create` or `Experiment.load` to create or load an experiment, and then call `Experiment.run` to run it.
    """

    n_steps: int
    logdir: str
    loggers: Collection[LoggerType]
    creation_timestamp: datetime

    @classmethod
    def load(cls, logdir: Path | str):
        """Load an experiment from a log directory."""
        json_file = cls.json_file(logdir)
        return cls.from_file(json_file, exact_type=True)

    @property
    def logpath(self):
        return Path(self.logdir)

    @property
    def experiment_file(self):
        return self.logpath / EXPERIMENT_FILENAME

    def replay_episode(self, run_seed: int, time_step: int, test_num: int, *, only_saved_actions: bool = False):
        """Replay the `test_num`th test episode at the `time_step`th test step from the `run_num`th run."""
        run = self.get_run(run_seed)
        assert run is not None
        return run.to_full().replay_episode(time_step, test_num, only_saved_actions)

    def move(self, new_logdir: Path):
        """Move an experiment to a new directory."""
        # Load the runs before moving the files, because we will not be able to load them after the move.
        runs = list(self.runs)
        # 1) move all files (with weights, logs, etc)
        shutil.move(self.logdir, new_logdir)
        # 2) update the experiment.json file with the new logdir
        self.logdir = new_logdir.as_posix()
        self.save()
        # 3) each rundir has to be overwritten with the new logdir
        for run in runs:
            run.rundir = (new_logdir / run.runpath.parts[-1]).as_posix()
            run.save()

    @staticmethod
    def json_file(logdir: str | Path):
        logdir = Path(logdir)
        return logdir / EXPERIMENT_FILENAME

    @overload
    def get_run(self, seed: int, /) -> LightRun | None: ...
    @overload
    def get_run(self, rundir: str, /) -> LightRun | None: ...

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
                yield LightRun[E, T].load(rundir)
            except FileNotFoundError:
                # Not a run directory
                pass
            except orjson.JSONDecodeError:
                # TODO: create a recovery mechanism for corrupt runs
                pass

    @staticmethod
    def is_experiment_directory(logdir: str | Path) -> bool:
        """Check if a directory is an experiment directory."""
        logdir = Path(logdir)
        return Experiment.json_file(logdir).exists()

    @classmethod
    def find_experiment_directory(cls, subdir: Path) -> Path | None:
        """Find the experiment directory containing a given subdirectory."""
        if cls.is_experiment_directory(subdir):
            return subdir
        # Root
        parent = subdir.parent
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

    def save(self):
        self.to_file(self.experiment_file)

    def delete(self):
        shutil.rmtree(self.logpath, ignore_errors=True)

    def get_tests_at(self, time_step: int):
        summaries = [run.get_test_episodes(time_step) for run in self.runs]
        return [summary for run_summaries in summaries for summary in run_summaries]

    def n_active_runs(self):
        return len([run for run in self.runs if run.is_running])

    def get_results_datasets(
        self, granularity: int, aggregate_by: "TickColumn" = "time_step", metrics: str | Collection[str] | None = None
    ):
        """
        Retrieve the datasets relative to the experiment.

        If provided, only computes the metrics provided in the `metrics` argument.
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        cols = ["ticks", *[cs.contains(m) for m in metrics]] if metrics is not None else pl.col("*")
        results = self.get_results(granularity, aggregate_by)
        datasets = list[Dataset]()
        for category, stats_df in results.items():
            try:
                stats_df = stats_df.select(cols).collect()
            except pl.exceptions.ColumnNotFoundError:
                continue  # The dataframe is empty
            if stats_df.is_empty():
                continue
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

    def get_results(self, granularity: int, aggregate_by: "TickColumn" = "time_step"):
        """
        Return the category-wise metrics aggregated by rounded step buckets, or elapsed-time buckets when wall-time mode is enabled.

        Example: if the time steps are [1, 2, 3, 4, 5] and the granularity is 2, the time steps will be rounded to [0, 2, 2, 4, 4], and the metrics will be averaged for each time step, resulting in a dataframe with time steps [0, 2, 4].
        """
        runs = list(self.runs)
        # if self.env.is_multi_objective:
        #     qvalues = stats.compute_qvalues([run.qvalues_data(self.test_interval) for run in runs], self.logdir, replace_inf, self.qvalue_infos)
        return {
            "Test": stats.compute_experiment_results([run.test_metrics for run in runs], aggregate_by, granularity),
            "Train": stats.compute_experiment_results([run.train_metrics for run in runs], aggregate_by, granularity),
            "Training data": stats.compute_experiment_results(
                [run.training_data for run in runs], aggregate_by, granularity
            ),
        }

    def get_test_results(self, granularity: int, aggregate_by: "TickColumn" = "time_step"):
        return stats.compute_experiment_results([run.test_metrics for run in self.runs], aggregate_by, granularity)

    def get_train_results(self, granularity: int, aggregate_by: "TickColumn" = "time_step"):
        return stats.compute_experiment_results([run.train_metrics for run in self.runs], aggregate_by, granularity)

    def get_training_data_results(self, granularity: int, aggregate_by: "TickColumn" = "time_step"):
        return stats.compute_experiment_results([run.training_data for run in self.runs], aggregate_by, granularity)

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


@dataclass
class Experiment[E: MARLEnv, T: Trainer](LightExperiment):
    trainer: T
    env: EnvConfig[E]
    test_env: EnvConfig[E]

    @classmethod
    def create(
        cls,
        env: EnvConfig[E],
        trainer: T,
        logdir: str | Literal["auto", "test", "tmp"] | Path = "tmp",
        n_steps: int = 1_000_000,
        test_env: EnvConfig[E] | None = None,
        loggers: Collection[LoggerType] = ("csv",),
    ):
        """
        Create a new experiment with the given parameters and save it to disk.

        **Parameters:**
        ----------
        - `env`: The environment configuration to train the agent on.
        - `trainer`: The trainer configuration to train the agent with.
        - `logdir`: The directory to save the experiment in. If "auto", the logdir is a combination of trainer name and environment name. If "test" or "tmp", any pre-existing experiment with the same name is overwritten.
        - `n_steps`: The number of training steps to run.
        - `test_env`: Environment configuration to test the trained agent against. Defaults to `deepcopy(env)`.
        - `loggers`: The loggers to use for the experiment.

        **Raises:**
        ------
        - `FileExistsError`: If the logdir already exists.
        """
        if logdir == "auto":
            logdir = Path("logs", f"{trainer.name}-{env.name}").as_posix()
        else:
            logdir = Path(logdir).as_posix()
            if not logdir.startswith("logs"):
                logdir = Path("logs", logdir).as_posix()
        logpath = Path(logdir)
        if logpath.parts[-1].lower() in ("test", "tmp"):
            logging.info(f"Discarding pre-existing experiment {logdir}.")
            shutil.rmtree(logpath, ignore_errors=True)
        if logpath.exists():
            # Do not allow to overwrite an existing experiment
            raise FileExistsError(f"Experiment directory {logpath} already exists.")
        if test_env is None:
            test_env = deepcopy(env)
        exp = Experiment(
            n_steps,
            logpath.as_posix(),
            loggers,
            datetime.now(),
            trainer,
            env,
            test_env,
        )
        exp.save()
        return exp

    def create_runs(
        self,
        seeds: int | Collection[int],
        n_tests: int,
        test_interval: int,
        save_weights: bool,
        save_actions: bool,
    ):
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        if self.test_env is None:
            self.test_env = self.env
        runs = [
            Run.create(
                seed,
                (self.logpath / f"run-{seed}").as_posix(),
                self.trainer,
                self.env,
                self.n_steps,
                self.test_env,
                test_interval=test_interval,
                n_tests=n_tests,
                loggers=self.loggers,
                save_weights=save_weights,
                save_actions=save_actions,
            )
            for seed in seeds
        ]
        # Deepcopy to prevent modifying the original configs references
        return deepcopy(runs)

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
        device: DeviceLike = "auto",
        render_tests: bool = False,
        n_jobs: int | Literal["auto"] = "auto",
        disabled_gpus: Collection[int] = (),
    ):
        """
        Train the Agent on the environment according to the experiment parameters.

        Parameters:
        ---------
        - `gpu_strategy`: Strategy to select the GPU to run the experiment on when `device` is set to "auto". If "group", fits as many runs as possible on a single GPU. If "scatter", scatters runs across GPUs according to their available memory.
        - `n_jobs`: Number of parallel jobs to run. If "auto", uses the number GPUs not disabled.
        """
        from marl.runners import parallel_run, sequential_run

        if n_jobs == "auto":
            n_jobs = torch.cuda.device_count() - len(disabled_gpus)
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        runs = self.create_runs(seeds, n_tests, test_interval, save_weights, save_actions)
        if n_jobs <= 1 or len(runs) <= 1:
            return sequential_run(runs, device, gpu_strategy, quiet, render_tests, disabled_gpus)
        return parallel_run(runs, n_jobs, device, gpu_strategy, render_tests, disabled_gpus, quiet)
