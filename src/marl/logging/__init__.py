from time import time
from .logger_interface import Logger
from .tensorboard_logger import TensorBoardLogger


def default() -> Logger:
    """Returns the default logger"""
    return TensorBoardLogger(f"logs/{time()}")
