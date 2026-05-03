from dataclasses import dataclass, field
from pathlib import Path

from marl.logging import Logger, LoggerType

from .config import Config


@dataclass
class LogConfig(Config[Logger]):
    logdir: str = "logs/test"
    loggers: list[LoggerType] = field(default_factory=lambda: ["csv"])

    @property
    def logpath(self):
        return Path(self.logdir)

    def make(self):
        from marl.logging import CSVLogger, MultiLogger, NeptuneLogger, TBLogger, WABLogger

        if self.logpath.parts[0] != "logs":
            self.logdir = Path("logs", self.logpath).as_posix()

        loggers = list[Logger]()
        for spec in set(self.loggers):
            if spec == "tensorboard":
                loggers.append(TBLogger(self.logpath))
            elif spec == "csv":
                loggers.append(CSVLogger(self.logpath))
            elif spec == "wandb":
                loggers.append(WABLogger(self.logpath))
            elif spec == "neptune":
                loggers.append(NeptuneLogger(self.logpath))
            elif spec == "sqlite":
                raise NotImplementedError("SQLite logger requires additional parameters. Use SQLiteLogger directly.")
            else:
                raise ValueError(f"Unknown logger type: {spec}")
        if len(loggers) == 1:
            return loggers[0]
        return MultiLogger(*loggers)
