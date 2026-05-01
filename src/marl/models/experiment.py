import os
import pickle
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import orjson
import torch
from marlenv.models import MARLEnv, Space

from marl import exceptions
from marl.logging import LogSpecs
from marl.models.replay_episode import LightEpisodeSummary
from marl.models.trainer import Trainer
from marl.runners.simple_runner import get_test_seed
from marl.utils import default_serialization, encode_b64_image, stats

from .replay_episode import ReplayEpisode
from .run import Run


@dataclass
class Experiment[A: Space]:
    logdir: str
    test_interval: int
    creation_timestamp: int
    trainer: Trainer
    env: MARLEnv[A]
    n_steps: int
    test_env: MARLEnv[A]
    save_weights: bool
    logger: LogSpecs
    save_actions: bool

    @staticmethod
    def create(
        env: MARLEnv[A],
        n_steps: int,
        logdir: str = "logs/tests",
        trainer: Trainer | None = None,
        test_interval: int = 5_000,
        test_env: MARLEnv[A] | None = None,
        logger: LogSpecs = "csv",
        save_weights: bool = False,
        save_actions: bool = True,
        *,
        replace_if_exists: bool = False,
    ):
        """
        Create a new experiment in the specified directory.

        - `trainer` defaults to `NoTrain` trainer if not provided
        - `agent` defaults to `trainer.make_agent()` if not provided
        - `test_env` defaults to a deep copy of `env` if not provided
        """
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        if os.path.basename(logdir).lower() in ("test", "tests", "debug"):
            shutil.rmtree(logdir, ignore_errors=True)
        if test_env is None:
            test_env = deepcopy(env)
        if not env.has_same_inouts(test_env):
            raise ValueError("The test environment must have the same inputs and outputs as the training environment.")
        if not logdir.startswith("logs"):
            logdir = os.path.join("logs", logdir)
        if trainer is None:
            from marl.training import NoTrain

            trainer = NoTrain(env)
        experiment = Experiment(
            logdir,
            trainer=trainer,
            env=env,
            n_steps=n_steps,
            test_interval=test_interval,
            creation_timestamp=int(time.time() * 1000),
            test_env=test_env,
            logger=logger,
            save_weights=save_weights,
            save_actions=save_actions,
        )
        try:
            if replace_if_exists and os.path.exists(logdir):
                shutil.rmtree(logdir, ignore_errors=True)
            experiment.save()
            return experiment
        except FileExistsError:
            raise exceptions.ExperimentAlreadyExistsException(logdir)

    @classmethod
    def load(cls, logdir: str):
        """Load an experiment from disk."""
        with open(os.path.join(logdir, "experiment.pkl"), "rb") as f:
            experiment: Experiment = pickle.load(f)
        return experiment

    def save(self):
        """Save the experiment to disk."""
        os.makedirs(self.logdir, exist_ok=False)
        with open(self.json_file(self.logdir), "wb") as f:
            f.write(orjson.dumps(self, default=default_serialization, option=orjson.OPT_SERIALIZE_NUMPY))
        with open(os.path.join(self.logdir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

    def _prepare_runs(self, seeds: Sequence[int]):
        """Create/load all the runs corresponding to the given seeds, and return them as a list."""
        runs = list[Run]()
        for seed in seeds:
            run = self.get_run_with_seed(seed)
            if run is None:
                run = Run.create(self.logdir, seed, self.logger)
            runs.append(run)
        return runs

    def run(
        self,
        seeds: int | Sequence[int] = 1,
        fill_strategy: Literal["scatter", "group"] = "group",
        quiet: bool = False,
        device: Literal["cpu", "auto"] | int = "auto",
        n_tests: int = 1,
        render_tests: bool = False,
        n_parallel: int = torch.cuda.device_count(),
        disabled_gpus: Sequence[int] = (),
    ):
        """Train the Agent on the environment according to the experiment parameters."""
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        runs = self._prepare_runs(seeds)
        if n_parallel <= 1 or len(runs) <= 1:
            from marl.runners import SequentialRunner

            runner = SequentialRunner(self)
            return runner.start(runs, device, fill_strategy, quiet, n_tests, render_tests, disabled_gpus)

        from marl.runners import ParallelRunner

        runner = ParallelRunner(self.logdir)
        return runner.start(
            runs,
            n_jobs=n_parallel,
            device=device,
            auto_device_strategy=fill_strategy,
            n_tests=n_tests,
            render_tests=render_tests,
            disabled_gpus=disabled_gpus,
            quiet=quiet,
        )

    def _make_replay_agent(self, run: Run, time_step: int, test_num: int, only_saved_actions: bool):
        from marl.agents import ReplayAgent

        if only_saved_actions:
            # This should fail if the actions file is not found
            actions = run.get_test_actions(time_step, test_num)
            return ReplayAgent.from_actions_only(actions)
        try:
            # This should **not** fail if the actions file is not found
            actions = run.get_test_actions(time_step, test_num)
            checkpoint_path = run.get_saved_algo_dir(time_step)
            return ReplayAgent.from_agent_and_actions(self.trainer.make_agent(), actions, checkpoint_path)
        except FileNotFoundError:
            pass
        try:
            return ReplayAgent.from_agent_only(self.trainer.make_agent(), run.get_saved_algo_dir(time_step))
        except FileNotFoundError:
            pass
        try:
            return ReplayAgent.from_actions_only(run.get_test_actions(time_step, test_num))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find any data to replay the episode for time step {time_step} and test number {test_num} in run with seed {run.seed}."
            )

    def replay_episode(self, run_num: int, time_step: int, test_num: int, *, only_saved_actions: bool = False) -> ReplayEpisode:
        """Replay the `test_num`th test episode at the `time_step`th test step from the `run_num`th run."""
        from marl.runners import seeded_rollout

        run = self.get_run(run_num)
        seed = get_test_seed(time_step, test_num)
        agent = self._make_replay_agent(run, time_step, test_num, only_saved_actions)
        episode, frames, detailed_actions = seeded_rollout(self.test_env, agent, seed, compute_frames=True)
        frames = [encode_b64_image(f) for f in frames]
        return ReplayEpisode(run.rundir, time_step, test_num, episode, frames, detailed_actions, self.test_env.action_space, agent)

    def move(self, new_logdir: str):
        """Move an experiment to a new directory."""
        shutil.move(self.logdir, new_logdir)
        self.logdir = new_logdir
        self.save()

    @staticmethod
    def json_file(logdir: str):
        return os.path.join(logdir, "experiment.json")

    def get_run(self, run_num: int):
        rundir = self.rundirs[run_num]
        return Run.load(rundir, self.logger)

    @property
    def runs(self):
        """All the runs related to the experiment."""
        for rundir in self.rundirs:
            yield Run.load(rundir, self.logger)

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

    @property
    def qvalue_infos(self):
        return (self.env.reward_space.labels, self.env.n_agents)

    def n_active_runs(self):
        return len([run for run in self.runs if run.is_running])

    def get_experiment_results(self, granularity: int | None = None, replace_inf=False, use_wall_time: bool = False):
        """Get all datasets of an experiment. If no qvalues were logged, the dataframe is empty"""
        if granularity is None:
            granularity = self.test_interval
        runs = list(self.runs)
        datasets = stats.compute_datasets(
            [run.test_metrics_aggregated(granularity, use_wall_time=use_wall_time) for run in runs],
            self.logdir,
            replace_inf,
            category="Test",
            use_wall_time=use_wall_time,
        )
        datasets += stats.compute_datasets(
            [run.train_metrics(granularity, use_wall_time=use_wall_time) for run in runs],
            self.logdir,
            replace_inf,
            category="Train",
            use_wall_time=use_wall_time,
        )
        datasets += stats.compute_datasets(
            [run.training_data(granularity, use_wall_time=use_wall_time) for run in runs],
            self.logdir,
            replace_inf,
            category="Other",
            use_wall_time=use_wall_time,
        )
        # if self.env.is_multi_objective:
        #     qvalues = stats.compute_qvalues([run.qvalues_data(self.test_interval) for run in runs], self.logdir, replace_inf, self.qvalue_infos)
        return datasets

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
