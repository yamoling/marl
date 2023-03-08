import logging
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from rlenv.models import Metrics, Measurement
from .logger_interface import Logger

class TensorBoardLogger(Logger):
    """Tensorboard logger"""
    def __init__(self, logdir: str) -> None:
        super().__init__(logdir)
        self.sw = SummaryWriter(self.logdir)
        self.to_print = []

    def log(self, tag: str, data: Metrics, time_step: int):
        match data:
            case Metrics():
                for s, measurement in data.items():
                    self.log(f"{tag}/{s}", measurement.value, time_step)
            case other:
                raise ValueError(f"Unsupported data type: {type(other)}")

    def print(self, tag: str, data: Metrics):
        for key, value in data.items():
            self.print(f"{tag}/{key}", value)
        self.flush()

    def flush(self, prefix: str=None):
        if prefix is not None:
            self.to_print.insert(0, prefix)
        logging.info("  \t".join(self.to_print))
        self.to_print = []
