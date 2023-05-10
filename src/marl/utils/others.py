import base64
import cv2
import torch
import numpy as np
from typing import Callable, TypeVar, Literal
import re
import socket
from dataclasses import dataclass
from contextlib import closing
   
def get_available_port(port=8000) -> int:
    MIN_PORT = 1024
    MAX_PORT = 65535
    port = min(max(port, MIN_PORT), MAX_PORT)
    for port in range(port, MAX_PORT):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(("0.0.0.0", port)) != 0:
                # Port is available
                return port
    raise RuntimeError("No available port found")

T = TypeVar("T")


@dataclass
class GPU:
    device: torch.device
    index: int
    name: str
    total_memory: int
    used_memory: int

    def __init__(self, index: int):
        self.device = torch.device(f"cuda:{index}")
        self.index = index
        self.name = torch.cuda.get_device_name(index)
        self.free_memory, self.total_memory = torch.cuda.mem_get_info(self.device)
        self.used_memory = self.total_memory - self.free_memory

    @property
    def memory_usage(self) -> float:
        """Memory usage between 0 and 1"""
        return self.used_memory / self.total_memory
    
    @property
    def utilization(self) -> float:
        return torch.cuda.utilization(self.index)
    
    def to_json(self) -> dict[str, float]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_usage": self.memory_usage,
            "utilization": self.utilization
        }

def list_gpus() -> list[GPU]:
    """List all available GPU devices"""
    return [GPU(i) for i in range(torch.cuda.device_count())]


def get_device(device: Literal["auto", "cuda", "cpu"]="auto") -> torch.device:
    """Get the given device"""
    if device == "auto" or device == "" or device is None:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        devices = list_gpus()
        # Order the GPUs by utilisation
        devices.sort(key=lambda g: g.utilization)
        for gpu in devices:
            if gpu.memory_usage < 0.9:
                return gpu.device
        raise RuntimeError("All GPU are full")
    return torch.device(device)


def defaults_to(value: T | None, default: Callable[[], T])  -> T:
    """
    Shortcut to retrieve a default value if the given one is None.
    """
    if value is not None:
        return value
    return default()
    

def alpha_num_order(string):
   """ Returns all numbers on 5 digits to let sort the string with numeric order.
   Ex: alphaNumOrder("a6b12.125")  ==> "a00006b00012.00125"
   """
   return ''.join([format(int(x), '08d') if x.isdigit()
                   else x for x in re.split(r'(\d+)', string)])


def encode_b64_image(image: np.ndarray) -> str:
    if image is None:
        return ""
    return base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("ascii")
