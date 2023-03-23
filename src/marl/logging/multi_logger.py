from rlenv.models import Metrics
from .logger_interface import Logger


class MultiLogger(Logger):
    def __init__(self, logdir: str, *loggers: Logger) -> None:
        super().__init__(logdir)
        self.loggers = loggers

    def log(self, tag: str, metrics: Metrics, time_step: int) -> None:
        for logger in self.loggers:
            logger.log(tag, metrics, time_step)

    def print(self, tag: str, metrics: Metrics) -> None:
        for logger in self.loggers:
            logger.print(tag, metrics)

    def flush(self, prefix: str|None = None) -> None:
        for logger in self.loggers:
            logger.flush(prefix)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
