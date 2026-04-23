from .logger import Logger, LogReader, TIME_STEP_COL, TIMESTAMP_COL
from .csv_logger import CSVLogger
from .wandb import WABLogger
from .neptune import NeptuneLogger
from .tensorboard import TBLogger
from .sql_logger import SQLiteLogger
from .multi_logger import MultiLogger
from typing import Literal, Sequence, TypeAlias


LogSpec: TypeAlias = Literal["tensorboard", "csv", "wandb", "neptune", "sqlite"]
LogSpecs: TypeAlias = LogSpec | Sequence[LogSpec]


def get_logger(logdir: str, specs: LogSpecs) -> Logger:
    loggers = list[Logger]()
    if isinstance(specs, str):
        specs = [specs]
    for spec in specs:
        if spec == "tensorboard":
            loggers.append(TBLogger(logdir))
        elif spec == "csv":
            loggers.append(CSVLogger(logdir))
        elif spec == "wandb":
            loggers.append(WABLogger(logdir))
        elif spec == "neptune":
            loggers.append(NeptuneLogger(logdir))
        elif spec == "sqlite":
            raise NotImplementedError("SQLite logger requires additional parameters. Use SQLiteLogger directly.")
        else:
            raise ValueError(f"Unknown log spec: {spec}")
    if len(loggers) == 1:
        return loggers[0]
    return MultiLogger(*loggers)


__all__ = [
    "Logger",
    "CSVLogger",
    "SQLiteLogger",
    "MultiLogger",
    "LogReader",
    "TIME_STEP_COL",
    "TIMESTAMP_COL",
    "WABLogger",
    "NeptuneLogger",
    "TBLogger",
    "LogSpec",
    "LogSpecs",
]
