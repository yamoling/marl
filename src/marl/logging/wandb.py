from typing import Optional, Any
import os
import dotenv
from marlenv import Episode, MARLEnv
from dataclasses import asdict
from marl.agents.agent import Agent
from marl.logging.logger import LogReader
from marl.models.trainer import Trainer
from .logger import Logger
import wandb


class WABLogger(Logger):
    def __init__(self, logdir: str, config: Optional[dict] = None):
        super().__init__(logdir)
        dotenv.load_dotenv()
        wandb.login()
        self.config = config
        project = os.getenv("WANDB_PROJECT", "marl")
        self._run = wandb.init(project=project, config=config, name=self.run_name)

    def log_params(self, trainer: Trainer, agent: Agent, env: MARLEnv, test_env: MARLEnv):
        self._run.config.update({"trainer": asdict(trainer)})
        self._run.config.update({"agent": asdict(agent)})
        self._run.config.update({"env": asdict(env)})
        self._run.config.update({"test_env": asdict(test_env)})

    @property
    def run_name(self):
        name = self.logdir
        if name.startswith("logs/"):
            name = name[5:]
        return name

    def log_test_episodes(self, episodes: list[Episode], time_step: int):
        for episode in episodes:
            metrics = {f"test/{k}": v for k, v in episode.metrics.items()}
            wandb.log(metrics, step=time_step)

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        if prefix is not None:
            data = {f"{prefix}{k}": v for k, v in data.items()}
        wandb.log(data, step=time_step)

    def __del__(self):
        wandb.finish()

    @staticmethod
    def reader(from_directory: str) -> LogReader:
        raise NotImplementedError("WandB logs cannot be read locally. Please visit https://wandb.ai to view your logs.")
