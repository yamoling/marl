import os
import time
import csv
import polars as pl
from .logger_interface import Logger


class CSVLogger(Logger):
    def __init__(self, filename: str, quiet: bool = False, flush_interval_sec: float = 30):
        Logger.__init__(self, os.path.dirname(filename), quiet)
        self.filename = filename
        self._file = open(filename, "w")
        self._writer = None
        self._flush_interval = flush_interval_sec
        self._next_flush = time.time() + flush_interval_sec

    def log(self, data: dict[str, float], time_step: int):
        if len(data) == 0:
            return
        now = time.time()
        data["timestamp_sec"] = now
        data["time_step"] = time_step
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=data.keys())
            self._writer.writeheader()
        try:
            self._writer.writerow(data)
        except ValueError:
            # Occurs when the header has changed compared to the previous data
            self._reformat(data)
            self._writer.writerow(data)
        if now >= self._next_flush:
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
        self._file.flush()
        self._file.close()
