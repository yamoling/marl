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


DeviceStr = Literal["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]


@serde
@dataclass
class GPU:
    index: int
    name: DeviceStr
    total_memory: int
    """Total memory (MB)"""
    used_memory: int
    """Used memory (MB)"""
    free_memory: int
    """Free memory (MB)"""
    memory_usage: float
    """Memory usage between 0 and 1"""
    utilization: float
    """Utilization between 0 and 1"""

    def __init__(self, index: int):
        self.name = f"cuda:{index}"  # type: ignore
        self.device = torch.device(index)
        self.index = index
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        self.free_memory = free_memory // (1024 * 1024)
        self.total_memory = total_memory // (1024 * 1024)
        self.used_memory = self.total_memory - self.free_memory
        self.memory_usage = self.used_memory / self.total_memory
        self.utilization = torch.cuda.utilization(self.device) / 100


def list_gpus() -> list[GPU]:
    """List all available GPU devices"""
    return [GPU(i) for i in range(torch.cuda.device_count())]


def get_device(
    device: DeviceStr = "auto",
    fit_strategy: Literal["fill", "conservative"] = "conservative",
    estimated_memory_MB: int = 0,
):
    """
    Get the given (GPU) device that fits the requirements.

    Arguments:
        - device: "auto" (default), "cuda" or "cpu"
        - fit_strategy:
            - "fill": Fit the process in the GPU that has the least free memory.
            - "conservative": Fit the process in the GPU that has the most free memory.
        - estimated_memory_MB: Estimated memory usage in MB.
    """
    if device != "auto":
        return torch.device(device)

    def fill(gpus: list[GPU], estimated_memory: int):
        gpus.sort(key=lambda gpu: gpu.free_memory)
        for gpu in gpus:
            if gpu.free_memory > estimated_memory:
                return gpu
        return None

    def conservative(gpus: list[GPU], estimated_memory: int):
        # The more utilization, the less the sorting score
        gpus.sort(key=lambda gpu: gpu.free_memory * (1.1 - gpu.utilization), reverse=True)
        for gpu in gpus:
            if gpu.free_memory > estimated_memory:
                return gpu
        return None

    if not torch.cuda.is_available():
        return torch.device("cpu")
    devices = list_gpus()
    match fit_strategy:
        case "fill":
            gpu = fill(devices, estimated_memory_MB)
        case "conservative":
            gpu = conservative(devices, estimated_memory_MB)
        case _:
            raise ValueError(f"Unknown fit strategy: {fit_strategy}. Choose 'fill' or 'conservative'")
    if gpu is None:
        return torch.device("cpu")
    return gpu.device


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
