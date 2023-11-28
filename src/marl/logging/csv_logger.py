import os
import time
from .logger_interface import Logger


class CSVLogger(Logger):
    def __init__(self, filename: str, quiet: bool = False, flush_interval_sec: float = 30):
        Logger.__init__(self, os.path.dirname(filename), quiet)
        self.filename = filename
        self._file = open(filename, "w")
        self._headers = list[str]()
        self._flush_interval = flush_interval_sec
        self._next_flush = time.time() + flush_interval_sec

    def log(self, data: dict[str, float]):
        now = time.time()
        data["timestamp_sec"] = now
        line = ""
        if len(self._headers) == 0:
            self._headers = list(data.keys())
            line = ",".join(self._headers) + "\n"
        line += ",".join(str(data.get(head)) for head in self._headers) + "\n"
        self._file.write(line)
        if now >= self._next_flush:
            self._file.flush()
            self._next_flush = now + self._flush_interval

    def close(self):
        self._file.flush()
        self._file.close()

    @property
    def test_path(self):
        return os.path.join(self.logdir, "test.csv")

    @property
    def train_path(self):
        return os.path.join(self.logdir, "train.csv")

    def __del__(self):
        self.close()
