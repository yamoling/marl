from .logger_interface import Logger
from .tensorboard_logger import TensorBoardLogger
from .ws_logger import WSLogger
from .multi_logger import MultiLogger


def default(path: str=None) -> Logger:
    """Returns the default logger"""
    from time import time
    if path is None:
        path = f"logs/{time()}"
    if not path.startswith("logs/"):
        import os
        path = os.path.join("logs", path)
    return TensorBoardLogger(path)
