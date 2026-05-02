import os
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import orjson

from marl.logging import LogSpecs

from .env_conf import EnvConf, env_conf_from_dict
from .trainer_config import TrainerConfig


@dataclass
class ExperimentConfig:
    logdir: str
    env: EnvConf
    trainer: TrainerConfig
    n_steps: int
    test_interval: int = 5_000
    test_env: EnvConf | None = None
    save_weights: bool = False
    logger: LogSpecs = "csv"
    save_actions: bool = True
    creation_timestamp: int | None = None

    @staticmethod
    def json_file(logdir: str) -> str:
        return os.path.join(logdir, "experiment.conf.json")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        env_data = data.get("env")
        trainer_data = data.get("trainer")
        test_env_data = data.get("test_env")
        if not isinstance(env_data, dict):
            raise ValueError("experiment configuration expects a nested 'env' object")
        if not isinstance(trainer_data, dict):
            raise ValueError("experiment configuration expects a nested 'trainer' object")

        test_env = None
        if isinstance(test_env_data, dict):
            test_env = env_conf_from_dict(test_env_data)

        return cls(
            logdir=str(data["logdir"]),
            env=env_conf_from_dict(env_data),
            trainer=trainer_conf_from_dict(trainer_data),
            n_steps=int(data["n_steps"]),
            test_interval=int(data.get("test_interval", 5_000)),
            test_env=test_env,
            save_weights=bool(data.get("save_weights", False)),
            logger=data.get("logger", "csv"),
            save_actions=bool(data.get("save_actions", True)),
            creation_timestamp=data.get("creation_timestamp"),
        )

    def save(self, path: str | None = None):
        if path is None:
            path = self.json_file(self.logdir)
        if os.path.isdir(path):
            path = self.json_file(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(orjson.dumps(self.to_dict(), option=orjson.OPT_SERIALIZE_NUMPY))

    @staticmethod
    def load(path_or_logdir: str) -> "ExperimentConfig":
        path = path_or_logdir
        if os.path.isdir(path_or_logdir):
            path = ExperimentConfig.json_file(path_or_logdir)
        with open(path, "rb") as f:
            return ExperimentConfig.from_dict(orjson.loads(f.read()))

    def build(self, *, logdir: str | None = None):
        from marl.models import Experiment

        runtime_logdir = self.logdir if logdir is None else logdir
        env = self.env.make()
        trainer = self.trainer.make(env)
        test_env = self.test_env.make() if self.test_env is not None else deepcopy(env)
        if self.creation_timestamp is None:
            creation_timestamp = int(time.time() * 1000)
        else:
            creation_timestamp = self.creation_timestamp
        experiment = Experiment(
            logdir=runtime_logdir,
            trainer=trainer,
            env=env,
            n_steps=self.n_steps,
            test_interval=self.test_interval,
            creation_timestamp=creation_timestamp,
            test_env=test_env,
            logger=self.logger,
            save_weights=self.save_weights,
            save_actions=self.save_actions,
        )
        return experiment

    def make_experiment(self, *, replace_if_exists: bool = False):
        experiment = self.build()
        if replace_if_exists and os.path.exists(experiment.logdir):
            shutil.rmtree(experiment.logdir, ignore_errors=True)
        experiment.save()
        self.save(experiment.logdir)
        return experiment
