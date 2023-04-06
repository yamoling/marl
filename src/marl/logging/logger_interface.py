from abc import ABC, abstractmethod
from typing import Literal
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
    def log(self, tag: Literal["train", "test"], data: Metrics, time_step: int):
        """Log the data."""

    def log_print(self, tag: Literal["train", "test"], data: Metrics, time_step: int):
        """Log to TensorBoard and add the data to the printing queue."""
        self.log(tag, data, time_step)
        if not self.quiet:
            self.print(tag, data)

    def print(self, tag: Literal["train", "test"], data: Metrics):
        """Add the data to the printing queue."""
        if not self.quiet:
            print(f"{tag}: {data}")

