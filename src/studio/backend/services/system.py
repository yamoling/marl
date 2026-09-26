"""CPU, RAM and GPU usage (same payload as the old `/system-specs` route)."""

from typing import Any

import psutil


def list_gpus() -> list[Any]:
    """`marl.utils.list_gpus` (imported lazily: it pulls torch). @ai-generated"""
    from marl.utils import list_gpus as _list_gpus

    return _list_gpus()


def system_info() -> dict[str, Any]:
    """`{cpu, ram, gpus}`: percentages in [0, 100], `marl.utils.gpu.GPU` objects. Blocking (nvidia-smi). @ai-generated"""
    return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent, "gpus": list_gpus()}
