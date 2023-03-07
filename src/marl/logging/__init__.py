from time import time
from .logger_interface import Logger
from .tensorboard_logger import TensorBoardLogger
from .ws_logger import WSLogger


def default(path: str=None) -> Logger:
    """Returns the default logger"""
    if path is None:
        path = f"logs/{time()}"
    if not path.startswith("logs/"):
        import os
        path = os.path.join("logs", path)
    return TensorBoardLogger(path)
