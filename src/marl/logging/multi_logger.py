from pprint import pprint
from .logger_interface import Logger


class MultiLogger(Logger):
    def __init__(self, logdir: str, *loggers: Logger, quiet=False) -> None:
        assert len(loggers) > 0, "At least one logger must be provided."
        super().__init__(logdir, quiet)
        self.loggers = loggers

    def log(self, tag: str, metrics: dict[str, float], time_step: int) -> None:
        for logger in self.loggers:
            logger.log(tag, metrics, time_step)

    def print(self, tag: str, metrics: dict[str, float]) -> None:
        print(f"\n{tag}")
        pprint(metrics)

    def __del__(self):
        for logger in self.loggers:
            logger.__del__()

    def close(self):
        for logger in self.loggers:
            logger.close()
