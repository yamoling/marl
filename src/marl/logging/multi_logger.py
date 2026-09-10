from typing import Any

from .logger import Logger


class MultiLogger(Logger):
    def __init__(self, *loggers: Logger) -> None:
        assert len(loggers) > 1, "At least two loggers must be provided."
        super().__init__(loggers[0].logdir)
        self.loggers = loggers

    def log(self, data: dict[str, Any], time_step: int, prefix: str | None = None):
        for logger in self.loggers:
            logger.log(data, time_step, prefix)

    def log_train(self, data: dict[str, Any], time_step: int):
        for logger in self.loggers:
            logger.log_train(data, time_step)

    def log_training_data(self, data: dict[str, Any], time_step: int):
        for logger in self.loggers:
            logger.log_training_data(data, time_step)

    def log_test_episodes(self, episodes, time_step: int, save_actions: bool = True):
        for logger in self.loggers:
            logger.log_test_episodes(episodes, time_step, save_actions)

    def close(self):
        for logger in self.loggers:
            logger.close()

    def reader(self):
        for logger in self.loggers:
            try:
                reader = logger.reader()
                return reader
            except NotImplementedError:
                continue
        raise NotImplementedError("None of the loggers have a reader implementation.")

    def __del__(self):
        for logger in self.loggers:
            del logger
