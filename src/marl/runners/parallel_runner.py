import signal
import time
from multiprocessing import get_context
from multiprocessing.pool import AsyncResult
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from torch import device

from marl.utils.gpu import get_device, get_gpu_processes, get_max_gpu_usage

from .simple_runner import SimpleRunner

if TYPE_CHECKING:
    from marl import Run


class ParallelRunner:
    def __init__(self, logdir: str):
        self.logdir = logdir

    def start(
        self,
        runs: "Sequence[Run]",
        n_jobs: int | None = None,
        device: int | device | str | Literal["auto", "cpu"] = "auto",
        auto_device_strategy: Literal["scatter", "group"] = "scatter",
        n_tests: int = 1,
        render_tests: bool = False,
        delay: int = 5,
    ):
        # If there are multiple GPUs, the first N_GPU runs will all try to fit on the same device.
        # For that reason, we sleep for delay seconds between each run to let the time to the
        # previous run to allocate memory on the GPU.
        n_gpus = torch.cuda.device_count()
        initial_pids = get_gpu_processes()
        estimated_gpu_memory = 0
        with get_context("spawn").Pool(n_jobs) as pool:
            handles = list[AsyncResult]()
            for run_num, run in enumerate(runs):
                run_device = get_device(device, auto_device_strategy, estimated_gpu_memory)
                # We want each child process to ignore the SIGINT signal so that if the user presses Ctrl+C, only the main process is killed and the children with it.
                original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                h = pool.apply_async(
                    _start_run,
                    kwds={
                        "logdir": self.logdir,
                        "seed": run.seed,
                        "device_index": run_device.index,
                        "n_tests": n_tests,
                        "quiet": (run_num != 0),
                        "render_tests": (run_num == 0) and render_tests,
                    },
                )
                # Restore the original SIGINT handler in the main process.
                signal.signal(signal.SIGINT, original_sigint_handler)
                handles.append(h)
                # If it is not the last process and there are multiple GPUs
                # then wait a bit to let the time to allocate the GPUs correctly.
                if run_device.index is not None and n_gpus > 1:
                    time.sleep(delay)
                    new_pids = get_gpu_processes() - initial_pids
                    estimated_gpu_memory = get_max_gpu_usage(new_pids)

            for h in handles:
                h.get()


def _start_run(logdir: str, seed: int, device_index: int | None, n_tests: int, quiet: bool, render_tests: bool):
    from marl import Experiment, Run

    exp = Experiment.load(logdir)
    run = exp.get_run_with_seed(seed)
    if run is None:
        run = Run.create(exp.logdir, seed, exp.logger)
    runner = SimpleRunner.from_experiment(exp, n_tests, quiet)
    if device_index is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_index)
    return runner.to(device).start(run, render_tests)
