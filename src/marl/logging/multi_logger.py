from typing import Any

from marlenv import MARLEnv

from marl.agents.agent import Agent
from marl.logging.logger import LogReader
from marl.models.trainer import Trainer
from .logger import Logger


class MultiLogger(Logger):
    def __init__(self, logdir: str, *loggers: Logger) -> None:
        assert len(loggers) > 0, "At least one logger must be provided."
        super().__init__(logdir)
        self.loggers = loggers

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        for logger in self.loggers:
            logger.log(data, time_step, prefix)

    def log_params(self, trainer: Trainer, agent: Agent, env: MARLEnv, test_env: MARLEnv):
        for logger in self.loggers:
            logger.log_params(trainer, agent, env, test_env)

    @staticmethod
    def reader(from_directory: str) -> LogReader:
        raise NotImplementedError("MultiLogger.reader is not implemented yet.")

    def __del__(self):
        for logger in self.loggers:
            del logger
