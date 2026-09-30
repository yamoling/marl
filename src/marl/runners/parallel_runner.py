from __future__ import annotations

import logging
import math
import multiprocessing as mp
import signal
import time
from collections import deque
from collections.abc import Collection
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_processes
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from marlenv import MARLEnv
from setproctitle import setproctitle

from marl.utils import DeviceLike
from marl.utils.gpu import GPUAllocationError, GPUAllocator, get_device, get_gpu_usage_by_pid, list_gpus

from ..models.trainer import Trainer
from .simple_runner import simple_run

if TYPE_CHECKING:
    from marl import Run

logger = logging.getLogger(__name__)

MEMORY_SAFETY_FACTOR = 1.1
"""Multiplier applied to the GPU memory usage observed for a run to estimate the usage of its experiment's runs."""


@contextmanager
def ignore_sigint():
    try:
        original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        # signal.signal can only be called from the main thread. If we're not in the main thread, we can't ignore SIGINT, but we also don't want to crash, so we just yield without changing the signal handler.
        logger.warning("Cannot ignore SIGINT in a non-main thread. SIGINT will not be ignored for this run.")
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)


def parallel_run[E: MARLEnv, T: Trainer](
    runs: Collection[Run[E, T]],
    n_jobs: int | None = None,
    device: DeviceLike = "auto",
    gpu_strategy: Literal["scatter", "group"] = "group",
    render_tests: bool = False,
    disabled_gpus: Collection[int] = (),
    quiet: bool = False,
    limit_torch_threads: int | Literal["auto"] | None = 1,
    device_affinity: int | None = None,
    *,
    initial_gpu_memory_mb: int = 1024,
    poll_interval_s: float = 3.0,
):
    """
    Train the given runs in at most `n_jobs` parallel worker processes.

    The runs may belong to different experiments. They share a single queue, so the runs of an experiment start
    as soon as a worker slot and a GPU are available, without waiting for the runs of the previous experiments.

    - `device`: "auto" places each run on a GPU with the `gpu_strategy` (see `_RunScheduler`); any other value
      (e.g. "cpu", "cuda:1" or 1) is used for every run.
    - `initial_gpu_memory_mb`: provisional GPU memory estimate of an experiment until the usage of one of its
      runs has been observed.
    - `poll_interval_s`: interval between two GPU memory measurements of the workers.

    Raises a `GPUAllocationError` with `device="auto"` if CUDA is unavailable or if the remaining runs fit on no
    GPU while no run is active. Runs are never moved to the CPU implicitly.

    @ai-edited
    """
    if n_jobs is None:
        n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs}")
    if device == "auto" and not torch.cuda.is_available():
        raise GPUAllocationError("device='auto' requires CUDA, but CUDA is not available. Pass device='cpu' to use the CPU.")
    scheduler = _RunScheduler(
        n_jobs=n_jobs,
        device=device,
        gpu_strategy=gpu_strategy,
        disabled_gpus=disabled_gpus,
        quiet=quiet,
        render_tests=render_tests,
        limit_torch_threads=limit_torch_threads,
        device_affinity=device_affinity,
        initial_gpu_memory_mb=initial_gpu_memory_mb,
        poll_interval_s=poll_interval_s,
    )
    try:
        scheduler.run(runs)
    except RuntimeError as e:
        if "__main__" in str(e):
            raise RuntimeError("""
This error occurred while spawning a worker process and is likely caused by not protecting the entry point with if __name__ == '__main__'.
Make sure to guard the entry of your program with the following:

if __name__ == '__main__':
    # Your code here
    experiment.run(...)
""") from e
        raise


@dataclass
class _Worker:
    key: int
    run: Run
    process: BaseProcess
    device: str
    experiment: Path
    bootstrap: bool
    """Whether this run was started to measure the GPU memory usage of an experiment without estimate."""
    max_memory_mb: int = 0
    last_sample_mb: int | None = None


class _RunScheduler:
    """
    Dispatch a shared queue of runs, possibly from several experiments, to at most `n_jobs` worker processes.

    Each run has its own spawned process, whose PID identifies its GPU memory usage in `nvidia-smi`.
    With `device="auto"`:
    - Every experiment directory has its own GPU memory estimate. It is set once the usage of a first
      (bootstrap) run is stable, and then raised to the largest usage observed across its runs, both
      multiplied by `MEMORY_SAFETY_FACTOR`. Until then, the other runs of that experiment are held back.
    - A run that fits on no GPU goes back to the end of the queue and the next run is tried.
    - If no queued run fits and no run is active, a `GPUAllocationError` is raised.

    @ai-generated
    """

    def __init__(
        self,
        n_jobs: int,
        device: DeviceLike,
        gpu_strategy: Literal["scatter", "group"],
        disabled_gpus: Collection[int],
        quiet: bool,
        render_tests: bool,
        limit_torch_threads: int | Literal["auto"] | None,
        device_affinity: int | None,
        initial_gpu_memory_mb: int,
        poll_interval_s: float,
    ):
        self.n_jobs = n_jobs
        self.disabled_gpus = disabled_gpus
        self.quiet = quiet
        self.render_tests = render_tests
        self.limit_torch_threads: int | Literal["auto"] | None = limit_torch_threads
        self.initial_gpu_memory_mb = initial_gpu_memory_mb
        self.poll_interval_s = poll_interval_s
        self.context = mp.get_context("spawn")
        self.explicit_device = None if device == "auto" else str(get_device(device))
        self.allocator = GPUAllocator(gpu_strategy, disabled_gpus, affinity=device_affinity) if device == "auto" else None
        self.estimates = dict[Path, int]()
        self.workers = list[_Worker]()
        self.n_started = 0
        self.last_sample_time = 0.0

    def run(self, runs: Collection[Run]):
        """Train all the runs, then return. @ai-generated"""
        queue = deque(runs)
        try:
            while len(queue) > 0 or len(self.workers) > 0:
                self._dispatch(queue)
                if len(self.workers) == 0:
                    raise GPUAllocationError(self._placement_error(queue))
                self._wait()
        finally:
            for worker in self.workers:
                logger.warning(f"Terminating {worker.run.rundir}")
                worker.process.terminate()
                worker.process.join(timeout=10)
                if self.allocator is not None:
                    self.allocator.release(worker.key)
            self.workers.clear()

    def _dispatch(self, queue: deque[Run]):
        """Try every queued run once, putting the runs that cannot start back at the end of the queue. @ai-generated"""
        for _ in range(len(queue)):
            if len(self.workers) >= self.n_jobs:
                return
            run = queue.popleft()
            if not self._try_start(run):
                queue.append(run)

    def _try_start(self, run: Run) -> bool:
        """Start the run on a device that can host it, and return whether it started. @ai-generated"""
        key = self.n_started
        experiment = run.runpath.parent.resolve()
        bootstrap = False
        if self.allocator is None:
            assert self.explicit_device is not None
            device = self.explicit_device
        else:
            bootstrap = experiment not in self.estimates
            if bootstrap and any(w.bootstrap and w.experiment == experiment for w in self.workers):
                return False
            index = self.allocator.acquire(key, self.estimates.get(experiment, self.initial_gpu_memory_mb))
            if index is None:
                return False
            device = f"cuda:{index}"
        # Only the first run may be verbose and render its tests.
        first = self.n_started == 0
        try:
            process = _spawn_worker(
                self.context,
                run,
                device,
                self.quiet if first else True,
                self.render_tests if first else False,
                self.limit_torch_threads,
            )
        except BaseException:
            if self.allocator is not None:
                self.allocator.release(key)
            raise
        self.workers.append(_Worker(key, run, process, device, experiment, bootstrap))
        self.n_started += 1
        logger.info(f"Started {run.rundir} on {device}")
        return True

    def _wait(self):
        """Wait until a worker finishes or the polling interval elapses, then update the workers. @ai-generated"""
        timeout = max(0.0, self.last_sample_time + self.poll_interval_s - time.monotonic())
        wait_for_processes([worker.process.sentinel for worker in self.workers], timeout=timeout)
        if time.monotonic() - self.last_sample_time >= self.poll_interval_s:
            self._sample_gpu_memory()
        self._reap()

    def _sample_gpu_memory(self):
        """Update the memory estimates and the pending reservations from the per-process usage. @ai-generated"""
        self.last_sample_time = time.monotonic()
        if self.allocator is None:
            return
        usage = get_gpu_usage_by_pid()
        for worker in self.workers:
            current = usage.get(worker.process.pid or -1, 0)
            worker.max_memory_mb = max(worker.max_memory_mb, current)
            if worker.experiment in self.estimates or (worker.bootstrap and current > 0 and current == worker.last_sample_mb):
                self._raise_estimate(worker.experiment, worker.max_memory_mb)
            worker.last_sample_mb = current
            estimate = self.estimates.get(worker.experiment, self.initial_gpu_memory_mb)
            self.allocator.set_pending(worker.key, estimate - current)

    def _raise_estimate(self, experiment: Path, observed_mb: int):
        """@ai-generated"""
        estimate = math.ceil(observed_mb * MEMORY_SAFETY_FACTOR)
        previous = self.estimates.get(experiment)
        if previous is None or estimate > previous:
            self.estimates[experiment] = estimate
            logger.info(f"GPU memory estimate of {experiment}: {estimate} MB (observed {observed_mb} MB)")

    def _reap(self):
        """Release the reservations of the finished workers and report failures. @ai-generated"""
        for worker in [w for w in self.workers if w.process.exitcode is not None]:
            worker.process.join()
            self.workers.remove(worker)
            if self.allocator is not None:
                self.allocator.release(worker.key)
            if worker.process.exitcode != 0:
                logger.error(f"Run {worker.run.rundir} failed with exit code {worker.process.exitcode}")
            else:
                logger.info(f"Run {worker.run.rundir} finished")
            if worker.bootstrap and worker.experiment not in self.estimates and worker.max_memory_mb > 0:
                self._raise_estimate(worker.experiment, worker.max_memory_mb)

    def _placement_error(self, queue: deque[Run]) -> str:
        """@ai-generated"""
        lines = [f"No GPU can host any of the {len(queue)} remaining run(s) and no run is active:"]
        for run in queue:
            experiment = run.runpath.parent.resolve()
            if experiment in self.estimates:
                lines.append(f"  - {run.rundir}: {self.estimates[experiment]} MB estimated")
            else:
                lines.append(f"  - {run.rundir}: {self.initial_gpu_memory_mb} MB provisional estimate")
        gpus = list_gpus(self.disabled_gpus)
        if len(gpus) == 0:
            lines.append("No GPU is allowed.")
        else:
            lines.append("Free memory of the allowed GPUs: " + ", ".join(f"cuda:{gpu.index} {gpu.free_memory} MB" for gpu in gpus))
        if len(self.disabled_gpus) > 0:
            lines.append(f"Disabled GPUs: {sorted(self.disabled_gpus)}")
        lines.append("Free some GPU memory, enable more GPUs, or pass device='cpu' to train on the CPU.")
        return "\n".join(lines)


def _spawn_worker(
    context: SpawnContext,
    run: Run,
    device: str,
    quiet: bool,
    render_tests: bool,
    limit_torch_threads: int | Literal["auto"] | None,
) -> BaseProcess:
    """Start a worker process that trains the run on the device. @ai-generated"""
    process = context.Process(
        target=_start_run,
        # Run.create() persists the complete run specification before it reaches the runner. Pass only its path
        # to the worker so that PyTorch does not create one shared-memory file descriptor per tensor storage.
        args=(run.rundir, device, quiet, render_tests, limit_torch_threads),
        name=f"worker: {run.rundir}",
    )
    # Workers inherit the ignored SIGINT, such that CTRL-C is captured by the parent process only.
    with ignore_sigint():
        process.start()
    return process


def _torch_thread_limit(limit_torch_threads: int | Literal["auto"] | None):
    if limit_torch_threads is None:
        return
    if limit_torch_threads != "auto":
        torch.set_num_threads(limit_torch_threads)
        torch.set_num_interop_threads(limit_torch_threads)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _start_run(
    rundir: str,
    device: str,
    quiet: bool,
    render_tests: bool,
    limit_torch_threads: int | Literal["auto"] | None,
):
    """Load a persisted run and train it on the device assigned by the parent. @ai-edited"""
    from ..models.run import Run

    run = Run.load(Path(rundir))
    setproctitle(f"worker: {run.rundir}")
    _torch_thread_limit(limit_torch_threads)
    logger.info(f"Selected device {device} for {run.rundir}")
    simple_run(run, quiet, render_tests, torch.device(device))
