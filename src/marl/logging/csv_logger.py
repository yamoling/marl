import os
import time
import csv
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
            self.rewrite_header(list(data.keys()))
            self._writer.writerow(data)
        if now >= self._next_flush:
            self._file.flush()
            self._next_flush = now + self._flush_interval

    def rewrite_header(self, headers: list[str]):
        self.close()
        assert self._writer is not None
        # Keep the same order for the headers
        new_headers = list(self._writer.fieldnames) + [h for h in headers if h not in self._writer.fieldnames]

        # Manually rewrite the first line (header)
        with open(self.filename, "r") as f:
            lines = f.readlines()
        lines[0] = ",".join(new_headers) + "\n"
        self._file = open(self.filename, "w")
        self._file.writelines(lines)

        # Reinitialize the writer
        self._writer = csv.DictWriter(self._file, fieldnames=new_headers)

    def close(self):
        self._file.flush()
        self._file.close()
