from abc import ABC, abstractmethod
from typing import Optional
from pprint import pprint
import numpy as np
import os
import cv2
import orjson
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
        self.logdir = logdir
        if os.path.exists(logdir):
            if os.path.basename(logdir).lower() in ["test", "debug"]:
                shutil.rmtree(logdir)
        os.makedirs(logdir, exist_ok=True)
        self.quiet = quiet

    def get_logdir(self, time_step: int) -> str:
        return os.path.join(self.logdir, str(time_step))

    @abstractmethod
    def log(self, data: dict[str, float], time_step: int):
        """Log the data."""

    def log_print(self, data: dict[str, float], time_step: int):
        """Log and print the data."""
        self.log(data, time_step)
        if not self.quiet:
            self.print(data)

    def print(self, data: dict[str, float]):
        """Add the data to the printing queue."""
        if not self.quiet:
            pprint(data)

    def log_as_json(self, object: object, time_step: int, name: Optional[str] = None):
        directory = self.get_logdir(time_step)
        if name is None:
            name = object.__class__.__name__
        filename = os.path.join(directory, f"{name}.json")
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(directory, f"{name}-{counter}.json")
            counter += 1
        with open(filename, "wb") as f:
            f.write(orjson.dumps(object, option=orjson.OPT_SERIALIZE_NUMPY))

    def log_image(self, image: np.ndarray, time_step: int, filename: str):
        directory = self.get_logdir(time_step)
        if not filename.endswith(".png") and not filename.endswith(".jpg"):
            filename += ".png"
        filename = os.path.join(directory, filename)
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(directory, f"image-{counter}.png")
            counter += 1
        cv2.imwrite(filename, image)

    @abstractmethod
    def close(self):
        """Close the logger"""
