import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

from .csv_logger import CSVLogger
from .logger import Logger, LogReader
from .multi_logger import MultiLogger
from .neptune import NeptuneLogger
from .progress import ProgressLogger
from .sql_logger import SQLiteLogger
from .tensorboard import TBLogger
from .wandb import WABLogger

type LoggerType = Literal["tensorboard", "csv", "wandb", "neptune", "sqlite", "progress"]
type LogSpecs = LoggerType | Sequence[LoggerType]
# Dataframe columns
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"
TICK_COL = "ticks"
TickColumn = Literal["time_step", "timestamp_sec"]

logger = logging.getLogger(__name__)


@lru_cache
def warn_once(msg: str):
    logger.warning(msg)


__all__ = [
    "TIMESTAMP_COL",
    "TIME_STEP_COL",
    "CSVLogger",
    "LogReader",
    "LogSpecs",
    "Logger",
    "LoggerType",
    "MultiLogger",
    "NeptuneLogger",
    "ProgressLogger",
    "SQLiteLogger",
    "TBLogger",
    "TickColumn",
    "WABLogger",
    "warn_once",
]
