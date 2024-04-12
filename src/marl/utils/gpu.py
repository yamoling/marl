from serde import serde
from typing import Literal
import torch
import subprocess
from dataclasses import dataclass


@serde
@dataclass
class GPU:
    index: int
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

    def __init__(self, index: int, total_memory: int, used_memory: int, free_memory: int, utilization: int):
        self.index = index
        self.total_memory = total_memory
        self.used_memory = used_memory
        self.free_memory = free_memory
        self.memory_usage = used_memory / total_memory
        self.utilization = utilization / 100


def list_gpus() -> list[GPU]:
    """List all available GPU devices"""
    try:
        cmd = "nvidia-smi  --format=csv,noheader,nounits --query-gpu=memory.total,memory.used,memory.free,utilization.gpu"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        return []
    res = []
    for i, line in enumerate(csv.split("\n")):
        total_memory, used_memory, free_memory, utilization = map(int, line.split(","))
        res.append(
            GPU(
                index=i,
                total_memory=total_memory,
                used_memory=used_memory,
                free_memory=free_memory,
                utilization=utilization,
            )
        )
    return res


def get_gpu_processes():
    try:
        cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
        return set(map(int, csv.split("\n")))
    except subprocess.CalledProcessError:
        # No GPU available
        return set[int]()
    except ValueError:
        # No processes
        return set[int]()


def get_max_gpu_usage(pids: set[int]):
    try:
        cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
        max_memory = 0
        for line in csv.split("\n"):
            pid, used_memory = map(int, line.split(","))
            if pid in pids:
                max_memory = max(max_memory, used_memory)
        return max_memory
    except subprocess.CalledProcessError:
        return 0


def get_device(
    device: Literal["auto", "cpu"] | int = "auto",
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
    return torch.device(gpu.index)
