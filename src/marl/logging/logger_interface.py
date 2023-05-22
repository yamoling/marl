from abc import ABC, abstractmethod
from pprint import pprint
import os
import shutil
import time
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
        self.quiet = quiet

    @abstractmethod
    def log(self, category: str, data: Metrics, time_step: int):
        """Log the data."""

    def log_print(self, category: str, data: Metrics, time_step: int):
        """Log to TensorBoard and add the data to the printing queue."""
        self.log(category, data, time_step)
        if not self.quiet:
            self.print(category, data)

    def print(self, category: str, data: Metrics):
        """Add the data to the printing queue."""
        if not self.quiet:
            pprint(f"{category}: {data}")
    
    @abstractmethod
    def close(self):
        """Close the logger"""

