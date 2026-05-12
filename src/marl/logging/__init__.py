import logging
from functools import lru_cache
from typing import Literal, Sequence, TypeAlias

from .csv_logger import CSVLogger
from .logger import Logger, LogReader
from .multi_logger import MultiLogger
from .neptune import NeptuneLogger
from .sql_logger import SQLiteLogger
from .tensorboard import TBLogger
from .wandb import WABLogger

LoggerType: TypeAlias = Literal["tensorboard", "csv", "wandb", "neptune", "sqlite"]
LogSpecs: TypeAlias = LoggerType | Sequence[LoggerType]
# Dataframe columns
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"
TICK_COL = "ticks"
TickColumn = Literal["time_step", "timestamp_sec"]


@lru_cache
def warn_once(msg: str):
    logging.warning(msg)


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
    "LoggerType",
    "LogSpecs",
    "TickColumn",
    "warn_once",
]
