import logging
import signal
import time
import multiprocessing as mp
from multiprocessing.pool import AsyncResult, Pool
from typing import TYPE_CHECKING, Literal, Sequence
from contextlib import contextmanager

import torch

from marl.utils.gpu import get_device, get_gpu_processes, get_gpu_usage_by_pid, scatter_plan

from .simple_runner import SimpleRunner

if TYPE_CHECKING:
    from marl import Run


mp.set_start_method("spawn", force=True)


@contextmanager
def ignore_sigint():
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
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
        delay: int = 5,
        disabled_gpus: Sequence[int] = (),
    ):
        initial_pids = get_gpu_processes()
        if n_jobs is None:
            n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1
        estimated_gpu_memory: int | None = None
        with Pool(n_jobs) as pool:
            handles = list[AsyncResult]()
            selected_device = get_device(device, auto_device_strategy)
            with ignore_sigint():
                h = pool.apply_async(
                    _start_run,
                    kwds={
                        "logdir": self.logdir,
                        "seed": runs[0].seed,
                        "device_type": selected_device.index,
                        "n_tests": n_tests,
                        "quiet": False,
                        "render_tests": render_tests,
                    },
                )
                handles.append(h)
            logging.info(f"Started first run on device {selected_device} to warm up and measure memory usage.")
            while estimated_gpu_memory is None:
                estimated_gpu_memory = _observe_new_process_memory(initial_pids, delay)
                logging.info(f"Estimated GPU memory usage of a single run: {estimated_gpu_memory} MB")
            preplanned_devices = []
            if auto_device_strategy == "scatter":
                preplanned_devices = scatter_plan(n_jobs - 1, estimated_gpu_memory, disabled_gpus)
                logging.info(f"Preplanned device assignments for scatter strategy: {preplanned_devices}")
            preplanned_devices += [device] * (len(runs) - 1 - len(preplanned_devices))
            logging.info(f"Final device assignments for all runs: {preplanned_devices}")
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
                        handles.pop(index)
                    except Exception as e:
                        print(f"Error in one of the runs: {e}")
                time.sleep(1)

    def _start_run(self):
        pass


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
    return runner.to(device).start(run, render_tests)


def _observe_new_process_memory(initial_pids: set[int], wait_s: float, poll_interval_s: float = 0.5):
    end_time = time.time() + max(wait_s, poll_interval_s)
    max_observed = 0
    prev_max_observed = None
    while time.time() < end_time or (max_observed != prev_max_observed):
        current_pids = get_gpu_processes() - initial_pids
        usage = get_gpu_usage_by_pid(current_pids)
        if len(usage) > 0:
            max_observed = max(max_observed, max(usage.values()))
            prev_max_observed = max_observed
        time.sleep(poll_interval_s)
    return max_observed if max_observed > 0 else None
