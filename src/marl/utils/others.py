import base64
import cv2
import torch
import numpy as np
from serde import serde
from typing import Callable, TypeVar, Literal
import re
from dataclasses import dataclass


def seed(seed_value: int):
    import torch
    import random
    import numpy as np

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
<<<<<<< HEAD
=======
    # Required for torch.bmm() to be deterministic
    torch.use_deterministic_algorithms(True)
>>>>>>> db50cbb15d296845231268ab17eb0bf8527aa5cc


T = TypeVar("T")


@serde
@dataclass
class GPU:
    index: int
    name: str
    total_memory: int
    used_memory: int
    utilization: float
    memory_usage: float

    def __init__(self, index: int):
        self.device = torch.device(f"cuda:{index}")
        self.index = index
        self.name = torch.cuda.get_device_name(index)
        self.free_memory, self.total_memory = torch.cuda.mem_get_info(self.device)
        self.used_memory = self.total_memory - self.free_memory
        self.utilization = torch.cuda.utilization(self.device)
        self.memory_usage = self.used_memory / self.total_memory
        """Memory usage between 0 and 1"""


def list_gpus() -> list[GPU]:
    """List all available GPU devices"""
    return [GPU(i) for i in range(torch.cuda.device_count())]


def get_device(device: Literal["auto", "cuda", "cpu"] = "auto") -> torch.device:
    """Get the given device"""
    if device == "auto" or device == "" or device is None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        devices = list_gpus()
        # Order the GPUs by utilisation * memory_usage
        devices.sort(key=lambda g: g.utilization * g.memory_usage)
        for gpu in devices:
            if gpu.memory_usage < 0.85:
                return gpu.device
        # Fallback to CPU if no GPU is available
        return torch.device("cpu")
    return torch.device(device)


def defaults_to(value: T | None, default: Callable[[], T]) -> T:
    """
    Shortcut to retrieve a default value if the given one is None.
    """
    if value is not None:
        return value
    return default()


def alpha_num_order(string):
    """Returns all numbers on 5 digits to let sort the string with numeric order.
    Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
    """
    return "".join([format(int(x), "08d") if x.isdigit() else x for x in re.split(r"(\d+)", string)])


def encode_b64_image(image: np.ndarray) -> str:
    if image is None:
        return ""
    return base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("ascii")  # type: ignore
