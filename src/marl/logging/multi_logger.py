from typing import Any

from marlenv import MARLEnv

from marl.models.agent import Agent
from marl.models.trainer import Trainer

from .logger import Logger


class MultiLogger(Logger):
    def __init__(self, *loggers: Logger) -> None:
        assert len(loggers) > 1, "At least two loggers must be provided."
        super().__init__(loggers[0].logdir)
        self.loggers = loggers

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        for logger in self.loggers:
            logger.log(data, time_step, prefix)

    def log_params(self, trainer: Trainer, agent: Agent, env: MARLEnv, test_env: MARLEnv):
        for logger in self.loggers:
            logger.log_params(trainer, agent, env, test_env)

    def reader(self):
        return self.loggers[0].reader()

    def __del__(self):
        for logger in self.loggers:
            del logger
