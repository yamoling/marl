import subprocess
from dataclasses import dataclass
import time
from typing import Literal, Sequence

import torch


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


def list_gpus(disabled_devices: Sequence[int] | None = None) -> list[GPU]:
    """List all available GPU devices except disabled ones"""
    if disabled_devices is None:
        disabled_devices = []
    try:
        cmd = "nvidia-smi --format=csv,noheader,nounits --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        return []
    res = []
    if len(csv) == 0:
        return res
    for line in csv.split("\n"):
        index, total_memory, used_memory, free_memory, utilization = map(int, line.split(","))
        if index in disabled_devices:
            continue
        res.append(
            GPU(
                index=index,
                total_memory=total_memory,
                used_memory=used_memory,
                free_memory=free_memory,
                utilization=utilization,
            )
        )
    return res


def get_gpu_processes() -> set[int]:
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


def scatter_plan(n_runs: int, required_memory_mb: int, disabled_gpus: Sequence[int] = ()):
    gpus = list_gpus(disabled_gpus)
    devices = list[int]()
    for _ in range(n_runs):
        min_gpu = None
        for i, gpu in enumerate(gpus):
            if gpu.free_memory > required_memory_mb:
                if min_gpu is None or gpu.free_memory > gpus[min_gpu].free_memory:
                    min_gpu = i
        if min_gpu is None:
            raise RuntimeError("No GPU can fit the required memory")
        devices.append(gpus[min_gpu].index)
    return devices


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
    except ValueError:
        # There is no process and int('') raises a ValueError
        return 0


def get_gpu_usage_by_pid(pids: set[int] | None = None) -> dict[int, int]:
    """Return per-process GPU memory usage (MB) for the provided pids."""
    try:
        cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
        if csv == "":
            return {}
        usage = dict[int, int]()
        for line in csv.split("\n"):
            pid, used_memory = map(int, line.split(","))
            if pids is not None and pid not in pids:
                continue
            usage[pid] = max(usage.get(pid, 0), used_memory)
        return usage
    except subprocess.CalledProcessError:
        return {}
    except ValueError:
        return {}


def select_gpu(
    fit_strategy: Literal["scatter", "group"] = "group",
    estimated_memory_MB: int = 0,
    disabled_devices: Sequence[int] | None = None,
):
    """Select a GPU that can fit the estimated memory requirements."""

    def grouped_fit(gpus: list[GPU], estimated_memory: int):
        gpus.sort(key=lambda gpu: gpu.free_memory)
        for gpu in gpus:
            if gpu.free_memory > estimated_memory:
                return gpu
        return None

    def scattered_fit(gpus: list[GPU], estimated_memory: int):
        # The more utilization, the less the sorting score.
        gpus.sort(key=lambda gpu: gpu.free_memory * (1.1 - gpu.utilization), reverse=True)
        for gpu in gpus:
            if gpu.free_memory > estimated_memory:
                return gpu

    devices = list_gpus(disabled_devices)
    match fit_strategy:
        case "group":
            return grouped_fit(devices, estimated_memory_MB)
        case "scatter":
            return scattered_fit(devices, estimated_memory_MB)
        case _:
            raise ValueError(f"Unknown fit strategy: {fit_strategy}. Choose 'group' or 'scatter'")


def wait_for_fitting_gpu(
    fit_strategy: Literal["scatter", "group"],
    estimated_memory_MB: int,
    disabled_devices: Sequence[int] | None = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
):
    """Wait until a GPU can fit the required memory and return it, else None on timeout."""
    start = time.time()
    while time.time() - start < timeout_s:
        gpu = select_gpu(fit_strategy, estimated_memory_MB, disabled_devices)
        if gpu is not None:
            return gpu
        time.sleep(poll_interval_s)
    return None


def get_device(
    device: Literal["auto", "cpu"] | int | torch.device | str = "auto",
    fit_strategy: Literal["scatter", "group"] = "group",
    estimated_memory_MB: int = 0,
    disabled_devices: Sequence[int] | None = None,
):
    """
    Get the given (GPU) device that fits the requirements.

    Arguments:
        - device: "auto" (default), "cuda" or "cpu"
        - fit_strategy:
            - "group": Fit the process in the GPU that has the least free memory (group all possible runs on a single GPU).
            - "scatter": Fit the process in the GPU that has the most free memory (scatter runs across all GPUs).
        - estimated_memory_MB: Estimated memory usage in MB.
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if device != "auto":
        if device == "cuda":
            return torch.device("cuda:0")
        return torch.device(device)

    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu = select_gpu(fit_strategy, estimated_memory_MB, disabled_devices)
    if gpu is None:
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu.index}")
