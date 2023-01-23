from abc import ABC, abstractmethod
from typing import Optional, Union, overload
import os
import shutil
import time
import logging
import sys
from rlenv.models import Measurement, Metrics


class Logger(ABC):
    """Logging interface"""
    def __init__(self, logdir: str) -> None:
        super().__init__()
        if logdir is None:
            logdir = f"{time.time()}"
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        self.logdir: str = logdir
        if os.path.exists(logdir):
            if os.path.basename(logdir).lower() in ["test", "debug"]:
                shutil.rmtree(logdir)
            else:
                move_destination = f"{logdir}-{time.time()}"
                logging.warning("The specified logdir alreasy exists, moving the old directory to %s", move_destination)
                shutil.move(logdir, move_destination)
        os.makedirs(logdir, exist_ok=True)
        file_handler = logging.FileHandler(filename=os.path.join(self.logdir, "training.log"))
        stdout_handler = logging.StreamHandler(sys.stderr)
        logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=[file_handler, stdout_handler])

    @overload
    def log(self, tag: str, data: Metrics, time_step: int):
        pass

    @overload
    def log(self, tag: str, data: Measurement, time_step: int):
        pass

    @overload
    def log(self, tag: str, data: float, time_step: int):
        pass

    @abstractmethod
    def log(self, tag, data, time_step):
        """Log the data to TensorBoard."""

    def log_print(self, tag: str, data: Union[Metrics, Measurement, float], time_step: int):
        """Log to TensorBoard and add the data to the printing queue."""
        self.log(tag, data, time_step)
        self.print(tag, data)

    @overload
    def print(self, tag: str, data: Measurement):
        pass

    @overload
    def print(self, tag: str, data: Metrics):
        pass

    @overload
    def print(self, tag: str, data: float):
        pass

    @abstractmethod
    def print(self, tag, data):
        """Add the data to the printing queue."""

    @abstractmethod
    def flush(self, prefix: str|None = None):
        """
        Flush the printing queue.
        If any, prepends the given prefix to the printing queue.
        """
