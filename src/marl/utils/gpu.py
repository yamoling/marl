import subprocess
import time
from collections.abc import Callable, Collection, Hashable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

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


def list_gpus(disabled_devices: Collection[int] | None = None) -> list[GPU]:
    """List all available GPU devices except disabled ones"""
    if disabled_devices is None:
        disabled_devices = []
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


def _break_tie(gpus: list[GPU], affinity: int | None) -> GPU:
    """
    Select one GPU among equally good candidates.

    Without affinity, the first candidate is returned, which makes concurrent processes pile up on
    the same device. With an affinity, the candidate at index `affinity % len(gpus)` is returned so
    that processes with different affinities (e.g. successive Optuna trials) spread over the tied
    devices.
    """
    if affinity is None:
        return gpus[0]
    return gpus[affinity % len(gpus)]


class GPUAllocationError(RuntimeError):
    """Raised when work must run on a GPU but no allowed GPU can host it."""


def pick_gpu(
    gpus: list[GPU],
    required_memory_mb: int,
    strategy: Literal["scatter", "group"],
    *,
    pending_memory_mb: Mapping[int, int] | None = None,
    active_workers: Mapping[int, int] | None = None,
    affinity: int | None = None,
) -> GPU | None:
    """
    Choose the GPU that best hosts `required_memory_mb` according to `strategy`, or None if none fits.

    - `pending_memory_mb`: memory promised to workers on each GPU that is not yet visible in the telemetry.
    - `active_workers`: number of workers that the caller already assigned to each GPU.

    "scatter" prefers the lowest compute load, i.e. the utilization plus one unit per active worker (which
    spreads workers that have not initialized CUDA yet), then the most available memory. "group" prefers the
    least available memory that still fits, so as to pack the workers on as few GPUs as possible. Ties are
    broken with the `affinity` (see `_break_tie`).

    @ai-generated
    """
    pending = pending_memory_mb or {}
    workers = active_workers or {}

    def available(gpu: GPU) -> int:
        return gpu.free_memory - pending.get(gpu.index, 0)

    score: Callable[[GPU], tuple[float, ...]]
    match strategy:
        case "scatter":
            score = lambda gpu: (gpu.utilization + workers.get(gpu.index, 0), -available(gpu))
        case "group":
            score = lambda gpu: (available(gpu),)
        case _:
            raise ValueError(f"Unknown fit strategy: {strategy}. Choose 'group' or 'scatter'")
    candidates = [gpu for gpu in gpus if available(gpu) > required_memory_mb]
    if len(candidates) == 0:
        return None
    best = min(map(score, candidates))
    return _break_tie([gpu for gpu in candidates if score(gpu) == best], affinity)


class GPUAllocator:
    """
    GPU reservations of the workers started by one launcher process.

    Every acquisition refreshes the GPU telemetry. A reservation holds the memory that its worker may still
    allocate: the launcher lowers it with `set_pending` once the worker's own usage shows up in the telemetry,
    so that this memory is not counted twice. Independent launchers do not share their reservations.
    """

    def __init__(
        self,
        strategy: Literal["scatter", "group"] = "scatter",
        disabled_gpus: Collection[int] = (),
        *,
        affinity: int | None = None,
    ):
        self.strategy: Literal["scatter", "group"] = strategy
        self.disabled_gpus = disabled_gpus
        self.affinity = affinity
        self._reservations: dict[Hashable, tuple[int, int]] = {}
        """Worker key -> (GPU index, pending memory in MB)"""

    def acquire(self, key: Hashable, required_memory_mb: int) -> int | None:
        """
        Reserve `required_memory_mb` on the best GPU for the worker `key` and return the GPU index,
        or None if no GPU can fit it.

        @ai-generated
        """
        if key in self._reservations:
            raise KeyError(f"Worker {key!r} already holds a GPU reservation")
        pending = dict[int, int]()
        workers = dict[int, int]()
        for index, memory in self._reservations.values():
            pending[index] = pending.get(index, 0) + memory
            workers[index] = workers.get(index, 0) + 1
        gpus = list_gpus(self.disabled_gpus)
        gpu = pick_gpu(gpus, required_memory_mb, self.strategy, pending_memory_mb=pending, active_workers=workers, affinity=self.affinity)
        if gpu is None:
            return None
        self._reservations[key] = (gpu.index, required_memory_mb)
        return gpu.index

    def set_pending(self, key: Hashable, pending_memory_mb: int):
        """Set the memory that the worker `key` may still allocate on its GPU. @ai-generated"""
        index, _ = self._reservations[key]
        self._reservations[key] = (index, max(0, pending_memory_mb))

    def release(self, key: Hashable):
        """Release the reservation of a finished worker. @ai-generated"""
        del self._reservations[key]


def scatter_plan(
    n_runs: int,
    required_memory_mb: int,
    disabled_gpus: Collection[int] = (),
    *,
    affinity: int | None = None,
):
    """
    Plan the placement of `n_runs` simultaneous runs with the "scatter" strategy of `pick_gpu`, accounting
    for the runs planned before. Use `GPUAllocator` instead when workers start and finish over time.

    @ai-edited
    """
    gpus = list_gpus(disabled_gpus)
    devices = list[int]()
    pending = dict[int, int]()
    workers = dict[int, int]()
    for _ in range(n_runs):
        selected = pick_gpu(gpus, required_memory_mb, "scatter", pending_memory_mb=pending, active_workers=workers, affinity=affinity)
        if selected is None:
            raise GPUAllocationError(f"Not enough GPUs to fit {n_runs} runs with {required_memory_mb} MB each.")
        devices.append(selected.index)
        pending[selected.index] = pending.get(selected.index, 0) + required_memory_mb
        workers[selected.index] = workers.get(selected.index, 0) + 1
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


def get_gpu_usage_by_pid() -> dict[int, int]:
    """Return per-process GPU memory usage (MB) for the provided pids."""
    try:
        cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        csv = subprocess.check_output(cmd, shell=True).decode().strip()
        if csv == "":
            return {}
        usage = dict[int, int]()
        for line in csv.split("\n"):
            pid, used_memory = map(int, line.split(","))
            usage[pid] = used_memory
        return usage
    except subprocess.CalledProcessError:
        return {}
    except ValueError:
        return {}


def select_gpu(
    fit_strategy: Literal["scatter", "group"] = "group",
    estimated_memory_MB: int = 0,
    disabled_devices: Collection[int] | None = None,
    *,
    affinity: int | None = None,
):
    """
    Select a GPU that can fit the estimated memory requirements.

    See `pick_gpu` for the fit strategies. Concurrent launchers should use a `GPUAllocator` instead,
    which accounts for the workers that they already started.

    @ai-edited
    """
    return pick_gpu(list_gpus(disabled_devices), estimated_memory_MB, fit_strategy, affinity=affinity)


def wait_for_fitting_gpu(
    fit_strategy: Literal["scatter", "group"],
    estimated_memory_MB: int,
    disabled_devices: Collection[int] | None = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
    *,
    affinity: int | None = None,
):
    """Wait until a GPU can fit the required memory and return it, else None on timeout."""
    start = time.time()
    while time.time() - start < timeout_s:
        gpu = select_gpu(fit_strategy, estimated_memory_MB, disabled_devices, affinity=affinity)
        if gpu is not None:
            return gpu
        time.sleep(poll_interval_s)
    return None


def get_device(
    device: DeviceLike | torch.device = "auto",
    fit_strategy: Literal["scatter", "group"] = "group",
    estimated_memory_MB: int = 0,
    disabled_devices: Collection[int] | None = None,
    *,
    affinity: int | None = None,
):
    """
    Get the given (GPU) device that fits the requirements.

    With `device="auto"`, a `GPUAllocationError` is raised when CUDA is unavailable or when no GPU fits:
    the CPU is only used when requested explicitly.

    Arguments:
        - device: "auto" (default), "cuda" or "cpu"
        - fit_strategy:
            - "group": Fit the process in the GPU that has the least free memory (group all possible runs on a single GPU).
            - "scatter": Prefer the least compute load, then the most free memory.
        - estimated_memory_MB: Estimated memory usage in MB.
        - affinity: Tie-breaker between equally good GPUs. When None (default), the first one is
        always selected. Otherwise, the GPU at index `affinity` (modulo the number of equally good
        GPUs) is selected, which spreads processes with distinct affinities across the devices.

    @ai-edited
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
        raise GPUAllocationError("device='auto' requires CUDA, but CUDA is not available. Pass device='cpu' to use the CPU.")
    gpu = select_gpu(fit_strategy, estimated_memory_MB, disabled_devices, affinity=affinity)
    if gpu is None:
        raise GPUAllocationError(
            f"No GPU has more than {estimated_memory_MB} MB of free memory (disabled GPUs: {list(disabled_devices or [])}). "
            "Free some GPU memory, enable more GPUs, or pass device='cpu' to use the CPU."
        )
    return torch.device(f"cuda:{gpu.index}")
