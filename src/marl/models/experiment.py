import os
import pathlib
import pickle
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Sequence, overload

import orjson
import torch
from marlenv.models import MARLEnv, Space

from marl import exceptions
from marl.logging import LogSpecs
from marl.models.replay_episode import LightEpisodeSummary
from marl.models.trainer import Trainer
from marl.runners.simple_runner import get_test_seed
from marl.utils import default_serialization, encode_b64_image, stats

from .agent import Agent
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
    logger: LogSpecs = "csv"

    @staticmethod
    def create(
        env: MARLEnv[A],
        n_steps: int,
        logdir: str = "logs/tests",
        trainer: Trainer | None = None,
        test_interval: int = 5_000,
        test_env: MARLEnv[A] | None = None,
        logger: LogSpecs = "csv",
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
        try:
            os.makedirs(logdir, exist_ok=False)
            experiment = Experiment(
                logdir,
                trainer=trainer,
                env=env,
                n_steps=n_steps,
                test_interval=test_interval,
                creation_timestamp=int(time.time() * 1000),
                test_env=test_env,
                logger=logger,
            )
            experiment.save()
            return experiment
        except FileExistsError:
            raise exceptions.ExperimentAlreadyExistsException(logdir)
        except Exception as e:
            # In case the experiment could not be created for another reason, do not create the experiment and remove its directory
            shutil.rmtree(logdir, ignore_errors=True)
            raise e

    @classmethod
    def load(cls, logdir: str):
        """Load an experiment from disk."""
        with open(os.path.join(logdir, "experiment.pkl"), "rb") as f:
            experiment: Experiment = pickle.load(f)
        return experiment

    def save(self):
        """Save the experiment to disk."""
        with open(self.json_file(self.logdir), "wb") as f:
            f.write(orjson.dumps(self, default=default_serialization, option=orjson.OPT_SERIALIZE_NUMPY))
        with open(os.path.join(self.logdir, "experiment.pkl"), "wb") as f:
            pickle.dump(self, f)

    def run(
        self,
        seeds: int | Sequence[int] = 0,
        fill_strategy: Literal["scatter", "group"] = "scatter",
        quiet: bool = False,
        device: Literal["cpu", "auto"] | int = "auto",
        n_tests: int = 1,
        render_tests: bool = False,
        n_parallel: int = torch.cuda.device_count(),
    ):
        """Train the Agent on the environment according to the experiment parameters."""
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        for seed in seeds:
            Run.create(self.logdir, seed, self.logger)
        if n_parallel <= 1:
            from marl.runners import SequentialRunner

            runner = SequentialRunner(self)
            return runner.start(device, fill_strategy, quiet, n_tests, render_tests)

        from marl.runners import ParallelRunner

        runner = ParallelRunner(self)
        return runner.start(
            n_jobs=n_parallel,
            device=device,
            auto_device_strategy=fill_strategy,
            n_tests=n_tests,
            render_tests=render_tests,
        )

    @overload
    def replay_episode(self, run_num: int, time_step: int, test_num: int, /) -> ReplayEpisode:
        """
        Replay the `test_num`th test episode at the `time_step`th test step from the `run_num`th run.

        Note that the actions are not re-evaluated from the agent but loaded from the `actions.json` file.
        """

    @overload
    def replay_episode(self, episode_folder: str, /) -> ReplayEpisode:
        """Replay the episode whose actions are saved in the given test folder."""

    def replay_episode(self, *args):
        match args:
            case (run_num, time_step, test_num):
                return self._replay_episode(run_num, time_step, test_num)
            case (episode_folder,):
                path = pathlib.Path(episode_folder)
                test_num = int(path.name)
                time_step = int(path.parent.name)
                rundir = str(path.parent.parent.parent)
                run_num = self.rundirs.index(rundir)
                return self._replay_episode(run_num, time_step, test_num)
            case _:
                raise ValueError("Invalid arguments")

    def _replay_episode(self, run_num: int, time_step: int, test_num: int):
        from marl.runners import seeded_rollout

        run = list(self.runs)[run_num]
        episode_folder = run.test_dir(time_step, test_num)
        # runner = self.create_runner()
        seed = get_test_seed(time_step, test_num)
        # actions = run.get_test_actions(time_step, test_num)
        agent = self.agent_at(time_step, run.seed)
        episode, frames, action_details = seeded_rollout(self.test_env, agent, seed, compute_frames=True)

        # episode = self.test_env.replay(actions, seed=seed)
        frames = [encode_b64_image(f) for f in frames]
        return ReplayEpisode(episode_folder, episode, frames, action_details)

    def agent_at(self, time_step: int, run_seed: int = 0) -> Agent:
        """Load the agent at a specific time step."""
        if time_step % self.test_interval != 0:
            raise ValueError(f"Time step must be a multiple of the test interval ({self.test_interval})")
        agent = self.trainer.make_agent()
        for run in self.runs:
            if run.seed == run_seed:
                agent.load(run.get_saved_algo_dir(time_step))
                return agent
        raise ValueError(f"No run with seed {run_seed} found in the experiment.")

    def move(self, new_logdir: str):
        """Move an experiment to a new directory."""
        shutil.move(self.logdir, new_logdir)
        self.logdir = new_logdir
        self.save()

    @staticmethod
    def json_file(logdir: str):
        return os.path.join(logdir, "experiment.json")

    @property
    def runs(self):
        for rundir in self.rundirs:
            yield Run(rundir, self.logger)

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
    def train_dir(self):
        return os.path.join(self.logdir, "train")

    @property
    def test_dir(self):
        return os.path.join(self.logdir, "test")

    @property
    def qvalue_infos(self):
        return (self.env.reward_space.labels, self.env.n_agents)

    def n_active_runs(self):
        return len([run for run in self.runs if run.is_running])

    def get_experiment_results(self, replace_inf=False):
        """Get all datasets of an experiment. If no qvalues were logged, the dataframe is empty"""
        runs = list(self.runs)
        datasets = stats.compute_datasets([run.test_metrics for run in runs], self.logdir, replace_inf, source="test", suffix=" [test]")
        datasets += stats.compute_datasets(
            [run.train_metrics(self.test_interval) for run in runs], self.logdir, replace_inf, source="train", suffix=" [train]"
        )
        datasets += stats.compute_datasets(
            [run.training_data(self.test_interval) for run in runs], self.logdir, replace_inf, source="training"
        )
        # qvalues = stats.compute_qvalues([run.qvalues_data(self.test_interval) for run in runs], self.logdir, replace_inf, self.qvalue_infos)
        return datasets, []

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
