from pprint import pprint
from .logger import Logger


class MultiLogger(Logger):
    def __init__(self, logdir: str, *loggers: Logger, quiet=False) -> None:
        assert len(loggers) > 0, "At least one logger must be provided."
        super().__init__(logdir, quiet)  # type: ignore
        self.loggers = loggers

    def log(self, metrics: dict[str, float], time_step: int):
        for logger in self.loggers:
            logger.log(metrics, time_step)  # type: ignore

    def print(self, tag: str, metrics: dict[str, float]):
        print(f"\n{tag}")
        pprint(metrics)

    def __del__(self):
        for logger in self.loggers:
            del logger
