from abc import ABC, abstractmethod
from typing import Optional, Union, overload
import os
import shutil
import time
import logging
import sys
from rlenv.models import Metrics


class Logger(ABC):
    """Logging interface"""
    def __init__(self, logdir: str, quiet=False) -> None:
        super().__init__()
        if logdir is None:
            logdir = f"{time.time()}"
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        self.logdir: str = logdir
        if os.path.exists(logdir):
            if os.path.basename(logdir).lower() in ["test", "debug"]:
                shutil.rmtree(logdir)
        os.makedirs(logdir, exist_ok=True)
        file_handler = logging.FileHandler(filename=os.path.join(self.logdir, "training.log"))
        stdout_handler = logging.StreamHandler(sys.stderr)
        logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=[file_handler, stdout_handler])
        self.quiet = quiet

    @abstractmethod
    def log(self, tag: str, data: Metrics, time_step: int):
        """Log the data."""

    def log_print(self, tag: str, data: Metrics, time_step: int):
        """Log to TensorBoard and add the data to the printing queue."""
        self.log(tag, data, time_step)
        if not self.quiet:
            self.print(tag, data)

    @abstractmethod
    def print(self, tag: str, data: Metrics):
        """Add the data to the printing queue."""

    @abstractmethod
    def flush(self, prefix: str|None = None):
        """
        Flush the printing queue.
        If any, prepends the given prefix to the printing queue.
        """
