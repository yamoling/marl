"""CPU, RAM and GPU usage (same payload as the old `/system-specs` route)."""

import subprocess
from dataclasses import dataclass
from typing import Any, Literal

import psutil

DeviceLike = Literal["cpu", "auto", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"] | int | str


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
    try:
        cmd = "nvidia-smi --format=csv,noheader,nounits --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
    except subprocess.CalledProcessError:
        return []
    if len(csv) == 0:
        return []
    res = []
    for line in csv.split("\n"):
        index, total_memory, used_memory, free_memory, utilization = map(int, line.split(","))
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


def system_info() -> dict[str, Any]:
    """`{cpu, ram, gpus}`: percentages in [0, 100], `marl.utils.gpu.GPU` objects. Blocking (nvidia-smi). @ai-generated"""
    return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "gpus": list_gpus()}
