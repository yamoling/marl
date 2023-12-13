import os
import threading
import time
from io import TextIOWrapper
from typing import Literal
from queue import Queue, Empty
from .logger_interface import Logger


class CSVLogger(Logger):
    def __init__(
        self, logdir: str, quiet: bool = False, flush_interval_sec: float = 30
    ):
        Logger.__init__(self, logdir, quiet)
        self._closed = False
        self._first = True
        self._headers = None
        self._next_flush = time.time() + flush_interval_sec

        if os.path.exists(self.test_path):
            self._resume_logging()
        self._train_file = open(self.train_path, "a")
        self._test_file = open(self.test_path, "a")

    def log(self, tag: Literal["train", "test"], data: dict, time_step: int):
        if self._closed:
            raise ValueError("Logger is closed")
        data["time_step"] = time_step
        data["timestamp_sec"] = time.time()
        if self._first:
            self._first = False
            self._headers = list(data.keys())
            str_headers = ",".join(self._headers) + "\n"
            self._train_file.write(str_headers)
            self._test_file.write(str_headers)
        line = ",".join(str(data.get(head)) for head in self._headers) + "\n"
        match tag:
            case "train":
                self._train_file.write(line)
            case "test":
                self._test_file.write(line)
            case other:
                raise ValueError(f"Unknown tag: {other}")
        now = time.time()
        if now >= self._next_flush:
            self._train_file.flush()
            self._test_file.flush()
            self._next_flush = now + 30

    def close(self):
        if not self._closed:
            self._closed = True
            self._train_file.close()
            self._test_file.close()

    def _resume_logging(self):
        with open(self.test_path, "r") as f:
            header = f.readline()
            if len(header) > 0:
                self._headers = header.strip().split(",")
                self._first = False

    @property
    def test_path(self):
        return os.path.join(self.logdir, "test.csv")

    @property
    def train_path(self):
        return os.path.join(self.logdir, "train.csv")

    def __del__(self):
        self.close()


class CSVLogger_THREADED(Logger):
    def __init__(self, logdir: str, quiet: bool = False, flush_interval: float = 120):
        Logger.__init__(self, logdir, quiet)
        self._train_queue = Queue()
        self._test_queue = Queue()
        self._train_file = open(os.path.join(self.logdir, "train.csv"), "w")
        self._test_file = open(os.path.join(self.logdir, "test.csv"), "w")
        self.train_thread = CSVWriterThread(
            self._train_queue, flush_interval, self._train_file
        )
        self.train_thread.start()
        CSVWriterThread(self._test_queue, flush_interval, self._test_file).start()
        self._closed = False

    def log(self, tag: Literal["train", "test"], data: dict, time_step: int):
        if self._closed:
            raise ValueError("Logger is closed")
        data["time_step"] = time_step
        data["timestamp_sec"] = time.time()
        match tag:
            case "train":
                self._train_queue.put(data)
            case "test":
                self._test_queue.put(data)
            case other:
                raise ValueError(f"Unknown tag: {other}")

    def close(self):
        if not self._closed:
            self._closed = True
            # Wait for all the messages to be written to the file.
            self._train_queue.join()
            self._train_file.close()
            self._test_queue.join()
            self._test_file.close()

    def __del__(self):
        print("Closing CSVLogger")
        self.close()
        print("Done closing CSVLogger")


class CSVWriterThread(threading.Thread):
    """Thread that processes asynchronous writes dict to CSV through a queue."""

    def __init__(self, queue: Queue, flush_interval: float, file: TextIOWrapper):
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._file = file
        self._flush_secs = flush_interval

    def _write_dict(self, dict: dict, headers: list[str]):
        str_data = ",".join([str(dict.get(h, "")) for h in headers]) + "\n"
        self._file.write(str_data)
        self._queue.task_done()

    def run(self):
        # Differentiate the first message from the rest to write the CSV headers.
        dict: dict = self._queue.get()
        if dict is None:
            self._queue.task_done()
            return
        headers = list(dict.keys())
        # Write the headers on the first message received
        self._file.write(",".join(headers) + "\n")
        self._write_dict(dict, headers)

        now = time.time()
        next_flush = now + self._flush_secs
        # Then, while the data is not None, write it to the file.
        while True:
            try:
                timeout = next_flush - now
                dict = self._queue.get(timeout=timeout)
                if dict is not None:
                    self._write_dict(dict, headers)
            except Empty:
                pass  # timeout
            now = time.time()
            if now >= next_flush:
                next_flush = now + self._flush_secs
                self._file.flush()
