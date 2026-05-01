import logging
import multiprocessing as mp
import signal
import time
from contextlib import contextmanager
from multiprocessing.pool import AsyncResult, Pool
from typing import TYPE_CHECKING, Literal, Sequence

import torch

from marl.utils.gpu import get_device, get_gpu_processes, get_gpu_usage_by_pid, scatter_plan

from .simple_runner import SimpleRunner

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


class ParallelRunner:
    def __init__(self, logdir: str):
        self.logdir = logdir

    def start(
        self,
        runs: "Sequence[Run]",
        n_jobs: int | None = None,
        device: int | str | Literal["auto", "cpu"] = "auto",
        auto_device_strategy: Literal["scatter", "group"] = "group",
        n_tests: int = 1,
        render_tests: bool = False,
        disabled_gpus: Sequence[int] = (),
        quiet: bool = False,
    ):
        if n_jobs is None:
            n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # use maxtasksperchild=1 such that CUDA memory is freed after each run.
        with mp.get_context("spawn").Pool(n_jobs, maxtasksperchild=1) as pool:
            estimated_gpu_memory, h = self._start_first_run(
                pool,
                runs[0],
                n_tests,
                render_tests,
                device,
                auto_device_strategy,
                disabled_gpus,
                quiet,
            )
            handles = [h]
            preplanned_devices = []
            if auto_device_strategy == "scatter":
                preplanned_devices = scatter_plan(n_jobs - 1, estimated_gpu_memory, disabled_gpus)
                logging.info(f"Preplanned device assignments for scatter strategy: {preplanned_devices}")
            preplanned_devices += [device] * (len(runs) - 1 - len(preplanned_devices))
            for i, run in enumerate(runs[1:]):
                with ignore_sigint():
                    h = pool.apply_async(
                        _start_run,
                        kwds={
                            "logdir": self.logdir,
                            "seed": run.seed,
                            "device_type": preplanned_devices[i],
                            "n_tests": n_tests,
                            "quiet": True,
                            "render_tests": render_tests,
                            "estimated_gpu_memory": estimated_gpu_memory,
                            "auto_device_strategy": auto_device_strategy,
                            "disabled_gpus": disabled_gpus,
                        },
                    )
                    handles.append(h)
            # Actively loop over the results to free up memory as soon as a run is finished
            while len(handles) > 0:
                ready_indices = [(i, h) for i, h in enumerate(handles) if h.ready()]
                for index, handle in reversed(ready_indices):
                    try:
                        handle.get(timeout=1)
                    except Exception as e:
                        print(f"Error in one of the runs: {e}")
                    finally:
                        # Always remove completed handles (including failures)
                        # to avoid waiting forever on an already-failed run.
                        handles.pop(index)
                time.sleep(1)

    def _start_first_run(
        self,
        pool: Pool,
        run: "Run",
        n_tests: int,
        render_tests: bool,
        device,
        auto_device_strategy: Literal["scatter", "group"],
        disabled_gpus: Sequence[int],
        quiet: bool,
    ):
        selected_device = get_device(device, auto_device_strategy, disabled_devices=disabled_gpus)
        initial_pids = get_gpu_processes()
        with ignore_sigint():
            h = pool.apply_async(
                _start_run,
                kwds={
                    "logdir": self.logdir,
                    "seed": run.seed,
                    "device_type": selected_device.index,
                    "n_tests": n_tests,
                    "quiet": quiet,
                    "render_tests": render_tests,
                    "estimated_gpu_memory": 0,
                    "auto_device_strategy": auto_device_strategy,
                    "disabled_gpus": disabled_gpus,
                },
            )
            logging.info(f"Started first run on device {selected_device} to warm up and measure memory usage.")
        estimated_gpu_memory = _estimate_required_gpu_memory(initial_pids, h)
        if estimated_gpu_memory is None:
            raise RuntimeError("Failed to estimate GPU memory usage of the first run.")
        logging.info(f"Estimated GPU memory usage of a single run: {estimated_gpu_memory} MB")
        return estimated_gpu_memory, h


def _start_run(
    logdir: str,
    seed: int,
    device_type: Literal["cpu", "auto"] | int | None,
    n_tests: int,
    quiet: bool,
    render_tests: bool,
    estimated_gpu_memory: int,
    auto_device_strategy: Literal["scatter", "group"],
    disabled_gpus: Sequence[int] = (),
):
    from marl import Experiment, Run

    exp = Experiment.load(logdir)
    run = exp.get_run_with_seed(seed)
    if run is None:
        run = Run.create(exp.logdir, seed, exp.logger)
    runner = SimpleRunner.from_experiment(exp, n_tests, quiet)
    match device_type:
        case int() | "cpu":
            device = torch.device(device_type)
        case None:
            device = torch.device("cpu")
        case "auto":
            device = get_device("auto", auto_device_strategy, estimated_gpu_memory, disabled_gpus)
        case other:
            raise ValueError(f"Invalid device_type: {other}")
    logging.info(f"Selected device {device} for {run.rundir}")
    return runner.to(device).start(run, render_tests)


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
    return max_observed
