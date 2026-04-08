import signal
import time
from copy import deepcopy
from multiprocessing.pool import AsyncResult, Pool
from typing import Literal, Sequence

import torch
from torch import device

from marl.utils.gpu import get_device, get_gpu_processes, get_max_gpu_usage

from .simple_runner import SimpleRunner


class ParallelRunner:
    def __init__(self, logdir: str):
        self.logdir = logdir

    def start(
        self,
        seeds: Sequence[int],
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
        n_gpus = torch.cuda.device_count() if device != "cpu" else 0
        initial_pids = get_gpu_processes()
        estimated_gpu_memory = 0
        with Pool(n_jobs) as pool:
            handles = list[AsyncResult]()
            for run_num, seed in enumerate(seeds):
                device = get_device(device, auto_device_strategy, estimated_gpu_memory)
                # We want each child process to ignore the SIGINT signal so that if the user presses Ctrl+C, only the main process is killed and the children with it.
                original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                h = pool.apply_async(
                    _start_run,
                    kwds={
                        "logdir": deepcopy(self.logdir),
                        "seed": seed,
                        "device_index": device.index,
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
                if n_gpus > 1 and run_num < len(seeds) - 1:
                    time.sleep(delay)
                    new_pids = get_gpu_processes() - initial_pids
                    estimated_gpu_memory = get_max_gpu_usage(new_pids)

            for h in handles:
                h.get()


def _start_run(logdir: str, seed: int, device_index: int | None, n_tests: int, quiet: bool, render_tests: bool):
    from marl import Experiment

    exp = Experiment.load(logdir)
    runner = SimpleRunner.from_experiment(exp, n_tests, quiet)
    if device_index is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_index)
    return runner.to(device).start(logdir, seed, exp.logger, render_tests)
