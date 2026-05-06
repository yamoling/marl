import logging
import multiprocessing as mp
import signal
import time
from contextlib import contextmanager
from multiprocessing.pool import AsyncResult, Pool
from typing import TYPE_CHECKING, Collection, Literal

import numpy.typing as npt
import torch
from marlenv import MARLEnv

from marl.utils.gpu import get_device, get_gpu_processes, get_gpu_usage_by_pid, scatter_plan

from .simple_runner import simple_run

if TYPE_CHECKING:
    from marl import Run


@contextmanager
def ignore_sigint():
    try:
        original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        # signal.signal can only be called from the main thread. If we're not in the main thread, we can't ignore SIGINT, but we also don't want to crash, so we just yield without changing the signal handler.
        logging.warning("Cannot ignore SIGINT in a non-main thread. SIGINT will not be ignored for this run.")
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)


def parallel_run[E: MARLEnv, T: npt.ArrayLike](
    runs: "Collection[Run[E, T]]",
    n_jobs: int | None = None,
    device: int | str | Literal["auto", "cpu"] = "auto",
    gpu_strategy: Literal["scatter", "group"] = "group",
    render_tests: bool = False,
    disabled_gpus: Collection[int] = (),
    quiet: bool = False,
):
    if n_jobs is None:
        n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    runs = list(runs)
    # use maxtasksperchild=1 such that CUDA memory is freed after each run.
    with mp.get_context("spawn").Pool(n_jobs, maxtasksperchild=1) as pool:
        # Start first run to measure GPU memory used
        pids = get_gpu_processes()
        handles = [submit(pool, runs[0], "auto", quiet, render_tests, 0, gpu_strategy, disabled_gpus)]
        estimated_gpu_memory = _estimate_required_gpu_memory(pids, handles[0])
        devices = []
        if gpu_strategy == "scatter":
            devices = scatter_plan(n_jobs - 1, estimated_gpu_memory, disabled_gpus)
            logging.info(f"Preplanned device assignments for scatter strategy: {devices}")
        devices += [device] * (len(runs) - 1 - len(devices))
        for run, device in zip(runs[1:], devices):
            handles.append(submit(pool, run, device, True, False, estimated_gpu_memory, gpu_strategy, disabled_gpus))
        # Actively loop over the results to free up memory as soon as a run is finished
        while len(handles) > 0:
            ready_indices = [(i, h) for i, h in enumerate(handles) if h.ready()]
            for index, handle in reversed(ready_indices):
                try:
                    handle.get(timeout=1)
                except Exception as e:
                    logging.error(f"Error in one of the runs: {e}", exc_info=e)
                finally:
                    # Always remove completed handles (including failures)
                    # to avoid waiting forever on an already-failed run.
                    handles.pop(index)
            time.sleep(1)


def submit(
    pool: Pool,
    run: "Run",
    device: Literal["cpu", "auto", "cuda"] | str | int | None,
    quiet: bool,
    render_tests: bool,
    estimated_gpu_memory: int,
    gpu_strategy: str,
    disabled_gpus: Collection[int],
):
    # Ignore sigint here such that CTRL-C is captured by the parent process
    with ignore_sigint():
        return pool.apply_async(
            _start_run,
            kwds={
                "run": run,
                "device_type": device,
                "quiet": quiet,
                "render_tests": render_tests,
                "estimated_gpu_memory": estimated_gpu_memory,
                "auto_device_strategy": gpu_strategy,
                "disabled_gpus": disabled_gpus,
            },
        )


def _start_run(
    run: "Run",
    device_type: Literal["cpu", "auto", "cuda"] | str | int | None,
    quiet: bool,
    render_tests: bool,
    estimated_gpu_memory: int,
    auto_device_strategy: Literal["scatter", "group"],
    disabled_gpus: Collection[int] = (),
):
    match device_type:
        case int() | "cpu":
            device = torch.device(device_type)
        case str(s) if s.startswith("cuda"):
            device = torch.device(s)
        case "auto" | None:
            device = get_device("auto", auto_device_strategy, estimated_gpu_memory, disabled_gpus)
        case other:
            raise ValueError(f"Invalid device_type: {other}")
    logging.info(f"Selected device {device} for {run.rundir}")
    return simple_run(run, quiet, render_tests, device)


def _estimate_required_gpu_memory(ignored_pids: set[int], run_0_handle: AsyncResult, poll_interval_s: float = 3.0):
    """
    Estimate the required GPU memory for a single run.

    This function monitors the GPU memory consumption of every process whose PID is not in the `ignored_pids` set (i.e. the processes that were already running before starting the first run). Once the same reading is observed twice in a row, it returns this value.
    """
    max_observed = None
    prev_max_observed = None
    while not run_0_handle.ready() and (max_observed is None or max_observed != prev_max_observed):
        time.sleep(poll_interval_s)
        prev_max_observed = max_observed
        usage = get_gpu_usage_by_pid()
        for pid, usage in usage.items():
            if pid not in ignored_pids:
                if max_observed is None or usage > max_observed:
                    max_observed = usage

    if max_observed is None:
        raise RuntimeError("Failed to estimate GPU memory usage of the first run.")
    logging.info(f"Estimated GPU memory usage of a single run: {max_observed} MB")
    return max_observed
