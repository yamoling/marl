from abc import ABC, abstractmethod
from pprint import pprint
import os
import shutil
import time


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
        self.quiet = quiet

    @abstractmethod
    def log(self, category: str, data: dict[str, float], time_step: int):
        """Log the data."""

    def log_print(self, category: str, data: dict[str, float], time_step: int):
        """Log and print the data."""
        self.log(category, data, time_step)
        if not self.quiet:
            self.print(category, data)

    def print(self, category: str, data: dict[str, float]):
        """Add the data to the printing queue."""
        if not self.quiet:
            pprint(f"{category}: {data}")

    @abstractmethod
    def close(self):
        """Close the logger"""
