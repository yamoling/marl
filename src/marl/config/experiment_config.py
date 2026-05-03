import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Literal

import torch
from marlenv import Space

from marl.logging import LoggerType
from marl.models.experiment import Experiment
from marl.models.run import Run

from .config import Config
from .env_config import EnvConfig
from .log_config import LogConfig
from .run_config import RunConfig
from .trainer_config import TrainerConfig


@dataclass
class ExperimentConfig[A: Space](Config[Experiment[A]]):
    env: EnvConfig[A]
    trainer: TrainerConfig
    n_steps: int = 1_000_000
    logdir: str = "logs/test"
    test_env: EnvConfig[A] | None = None
    """Environment configuration to test the trained agent against. Defaults to `self.env`."""
    loggers: list[LoggerType] = field(default_factory=lambda: ["csv"])

    @property
    def logpath(self):
        return Path(self.logdir)

    @property
    def config_path(self):
        return self.logpath / "experiment.json"

    @property
    def may_overwrite(self):
        """Whether a previous existing experiment with the same name may be overwritten or not."""
        return self.logpath.parts[1] in ("test", "tests", "debug")

    def get_run_with_seed(self, seed: int):
        for run in self.runs:
            if run.seed == seed:
                return run
        return None

    def create_runs(self, seeds: int | Collection[int], test_interval: int, save_weights: bool, save_actions: bool):
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        if self.test_env is None:
            self.test_env = self.env
        for seed in seeds:
            # Deepcopy to prevent modifying the original config
            yield deepcopy(
                RunConfig(
                    seed,
                    self.logpath / f"run-{seed}",
                    self.trainer,
                    self.env,
                    self.test_env,
                    self.n_steps,
                    test_interval,
                    LogConfig(self.logdir, self.loggers),
                    save_weights,
                    save_actions,
                )
            )

    def run(
        self,
        seeds: int | Collection[int] = 1,
        fill_strategy: Literal["scatter", "group"] = "group",
        quiet: bool = False,
        device: Literal["cpu", "auto"] | int = "auto",
        n_tests: int = 1,
        render_tests: bool = False,
        n_parallel: int = torch.cuda.device_count(),
        disabled_gpus: Collection[int] = (),
    ):
        pass

    def save(self):
        if self.may_overwrite:
            shutil.rmtree(self.logdir, ignore_errors=True)
        os.makedirs(self.logdir)
        self.to_file(self.config_path)

    def make(self):
        logger = self.logger.make()
        env = self.env.make()
        trainer = self.trainer.make()
        raise NotImplementedError()

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
