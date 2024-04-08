import base64
import cv2
import torch
import numpy as np
from serde import serde
from typing import Callable, Optional, TypeVar, Literal
import re
from dataclasses import dataclass
from rlenv import RLEnv


def seed(seed_value: int, env: Optional[RLEnv] = None):
    import torch
    import random
    import numpy as np

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if env is not None:
        env.seed(seed_value)


@serde
@dataclass
class GPU:
    index: int
    name: str
    total_memory: int
    used_memory: int
    utilization: float
    memory_usage: float
    """Memory usage between 0 and 1"""

    def __init__(self, index: int):
        self.device = torch.device(f"cuda:{index}")
        self.index = index
        self.name = torch.cuda.get_device_name(index)
        self.free_memory, self.total_memory = torch.cuda.mem_get_info(self.device)
        self.used_memory = self.total_memory - self.free_memory
        self.utilization = torch.cuda.utilization(self.device)
        self.memory_usage = self.used_memory / self.total_memory


def list_gpus() -> list[GPU]:
    """List all available GPU devices"""
    return [GPU(i) for i in range(torch.cuda.device_count())]


def get_device(
    device: Literal["auto", "cuda", "cpu"] = "auto",
    fit_strategy: Literal["fill", "conservative"] = "conservative",
    estimated_memory_GB: int = 0,
) -> torch.device:
    """
    Get the given (GPU) device that fits the requirements.

    Arguments:
        - device: "auto" (default), "cuda" or "cpu"
        - fit_strategy:
            - "fill": Fit the process in the GPU that has the least free memory.
            - "conservative": Fit the process in the GPU that has the most free memory.
        - estimated_memory_GB: Estimated memory usage in GB.
    """

    def fill(gpus: list[GPU], estimated_memory: int):
        gpus.sort(key=lambda gpu: gpu.free_memory)
        for gpu in gpus:
            if gpu.free_memory > estimated_memory:
                return gpu.device
        return None

    def conservative(gpus: list[GPU], estimated_memory: int):
        gpus.sort(key=lambda gpu: gpu.free_memory, reverse=True)
        for gpu in gpus:
            if gpu.free_memory > estimated_memory:
                return gpu.device
        return None

    if device == "auto" or device == "" or device is None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        devices = list_gpus()
        match fit_strategy:
            case "fill":
                gpu = fill(devices, estimated_memory_GB)
            case "conservative":
                gpu = conservative(devices, estimated_memory_GB)
            case _:
                raise ValueError(f"Unknown fit strategy: {fit_strategy}. Choose 'fill' or 'conservative'")
        if gpu is None:
            return torch.device("cpu")
        return gpu
    return torch.device(device)


T = TypeVar("T")


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
