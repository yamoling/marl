import logging
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from .logger_interface import Logger


class TensorBoardLogger(Logger):
    """Tensorboard logger"""

    def __init__(self, logdir: str, quiet=False) -> None:
        super().__init__(logdir, quiet)
        self.sw = SummaryWriter(self.logdir)
        self.to_print = []

    def log(self, tag: str, data: dict | float, time_step: int):
        match data:
            case float() | int():
                self.sw.add_scalar(tag, data, time_step)
            case dict():
                for s, value in data.items():
                    self.log(f"{tag}/{s}", value, time_step)
            case other:
                raise ValueError(f"Unsupported data type: {type(other)}")

    def print(self, tag: str, data: Union[dict, float]):
        if isinstance(data, dict):
            for key, value in data.items():
                self.print(f"{tag}/{key}", value)
            self.flush()
        else:
            self.to_print.append(f"{tag} {data}")

    def flush(self, prefix: str = None):
        if prefix is not None:
            self.to_print.insert(0, prefix)
        logging.info("  \t".join(self.to_print))
        self.to_print = []

    def __del__(self):
        self.flush()
        self.sw.close()

    def close(self):
        self.flush()
        self.sw.close()
