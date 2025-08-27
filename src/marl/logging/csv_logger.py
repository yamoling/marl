import os
import time
import csv
import polars as pl
from typing import Any
from .logger import Logger, LogReader, LogWriter

QVALUES = "qvalues.csv"
TRAIN = "train.csv"
TEST = "test.csv"
TRAINING_DATA = "training_data.csv"
ACTIONS = "actions.json"
PID = "pid"

# Dataframe columns
TIME_STEP_COL = "time_step"
TIMESTAMP_COL = "timestamp_sec"


class CSVWriter(LogWriter):
    def __init__(self, filename: str, quiet: bool = False, flush_interval_sec: float = 30):
        super().__init__(filename, quiet)
        self._file = None
        self._writer = None
        self._flush_interval = flush_interval_sec
        self._next_flush = time.time() + flush_interval_sec
        self._schema = {
            TIME_STEP_COL: "int",
            TIMESTAMP_COL: "float",
        }

    def log(self, data: dict[str, Any], time_step: int):
        if len(data) == 0:
            return
        now = time.time()
        data["timestamp_sec"] = now
        data["time_step"] = time_step
        if self._writer is None:
            if os.path.exists(self.filename):
                self._file = open(self.filename, "a")
            else:
                self._file = open(self.filename, "w")
            self._writer = csv.DictWriter(self._file, fieldnames=data.keys())
            self._writer.writeheader()
        try:
            self._writer.writerow(data)
        except ValueError:
            # Occurs when the header has changed compared to the previous data
            self._reformat(data)
            self._writer.writerow(data)
        if now >= self._next_flush:
            assert self._file is not None
            self._file.flush()
            self._next_flush = now + self._flush_interval

    def _reformat(self, data: dict[str, float]):
        """
        When trying to write a data item whose columns do not match the header, we need to
        re-write the while. We choose to pad the columns with None values.

        Note: this is costly if reformatting happens when the file is already large.
        """
        self.close()
        df = pl.read_csv(self.filename)
        new_headers = set(data.keys()) - set(df.columns)
        df = df.with_columns([pl.lit(None).alias(h) for h in new_headers])
        df.write_csv(self.filename)

        self._file = open(self.filename, "a")
        self._writer = csv.DictWriter(self._file, fieldnames=df.columns)

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()


class CSVLogReader(LogReader):
    def __init__(self, logdir: str):
        self.test_filename = os.path.join(logdir, TEST)
        self.train_filename = os.path.join(logdir, TRAIN)
        self.training_data_filename = os.path.join(logdir, TRAINING_DATA)

    def _read(self, filename: str) -> pl.DataFrame:
        try:
            # With SMAC, there are sometimes episodes that are not finished and that produce
            # None values for some metrics. We ignore these episodes.
            return pl.read_csv(filename, ignore_errors=True)
        except (pl.exceptions.NoDataError, FileNotFoundError):
            return pl.DataFrame()

    @property
    def test_metrics(self) -> pl.DataFrame:
        return self._read(self.test_filename)

    @property
    def train_metrics(self) -> pl.DataFrame:
        return self._read(self.train_filename)

    @property
    def training_data(self) -> pl.DataFrame:
        return self._read(self.training_data_filename)


class CSVLogger(Logger):
    def __init__(self, logdir: str, quiet: bool = False, flush_interval_sec: float = 30):
        test = CSVWriter(os.path.join(logdir, TEST), quiet, flush_interval_sec)
        train = CSVWriter(os.path.join(logdir, TRAIN), quiet, flush_interval_sec)
        training_data = CSVWriter(os.path.join(logdir, TRAINING_DATA), quiet, flush_interval_sec)

        super().__init__(logdir, quiet, test, train, training_data)

    @staticmethod
    def reader(from_directory: str) -> "CSVLogReader":
        return CSVLogReader(from_directory)
