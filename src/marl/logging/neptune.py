import os
from dataclasses import asdict
from typing import Any

import dotenv
import neptune
from marlenv import MARLEnv

from marl.agents.agent import Agent
from marl.logging.logger import LogReader
from marl.models.trainer import Trainer

from .logger import Logger


class NeptuneLogger(Logger):
    def __init__(self, logdir: str):
        super().__init__(logdir)

        dotenv.load_dotenv()
        project = os.getenv("NEPTUNE_PROJECT", "marl")
        api_key = os.getenv("NEPTUNE_API_TOKEN")
        self.run = neptune.init_run(project=project, api_token=api_key, custom_run_id=logdir)

    def log_params(self, trainer: Trainer, agent: Agent, env: MARLEnv, test_env: MARLEnv):
        self.run["config/trainer"] = asdict(trainer)
        self.run["config/agent"] = asdict(agent)
        self.run["config/env"] = asdict(env)
        self.run["config/test_env"] = asdict(test_env)

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        for k, v in data.items():
            if prefix is not None:
                k = f"{prefix}{k}"
            self.run[k].append(v, step=time_step)

    def __del__(self):
        self.run.stop()

    @staticmethod
    def reader(from_directory: str) -> LogReader:
        raise NotImplementedError("Neptune logs cannot be read locally. Please visit https://app.neptune.ai to view your logs.")
