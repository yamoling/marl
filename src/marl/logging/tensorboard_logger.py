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

    def log(self, tag: str, data: Union[Metrics, Measurement, float], time_step: int):
        if isinstance(data, (float, int)):
            self.sw.add_scalar(tag, data, time_step)
        elif isinstance(data, Measurement):
            self.sw.add_scalar(tag, data.value, time_step)
        elif isinstance(data, Metrics):
            for s, value in data.items():
                self.log(f"{tag}/{s}", value, time_step)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def print(self, tag: str, data: Union[Measurement, Metrics, float]):
        if isinstance(data, Metrics):
            for key, value in data.items():
                self.print(f"{tag}/{key}", value)
            self.flush()
        else:
            self.to_print.append(f"{tag} {data}")

    def flush(self, prefix: str=None):
        if prefix is not None:
            self.to_print.insert(0, prefix)
        logging.info("  \t".join(self.to_print))
        self.to_print = []

    def __del__(self):
        # self.sw.close()
        pass
