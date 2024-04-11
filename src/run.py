from typing import Callable, Literal, Optional
import torch
import polars as pl
import subprocess
import typed_argparse as tap
import multiprocessing as mp

import time

from marl.utils.others import DeviceStr


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    device: DeviceStr = tap.arg(default="auto")
    estimated_memory_MB: int = tap.arg(default=3_000, help="Estimated memory in GB for the 'auto' device")
    gpu_strategy: Literal["fill", "conservative"] = tap.arg(default="conservative")


def get_memory_usage_by_pid(pid: int):
    command = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"Failed to get memory usage: {result.stderr}")
    total = 0
    for line in result.stdout.decode().split("\n"):
        if line.strip() == "":
            continue
        key, usage = line.split(",")
        if int(key) == pid:
            total += int(usage)
    return total


def main(args: Arguments):
    # Import in the function to quicken the startup time
    import marl

    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    experiment = marl.Experiment.load(args.logdir)
    with mp.Pool(args.n_runs) as pool:
        for run_num in range(args.n_runs):
            pool.apply_async(
                experiment.run,
                kwds={
                    "self": experiment,
                    "seed": args.seed + run_num,
                    "fill_strategy": "conservative",
                    "required_memory_MB": args.estimated_memory_MB,
                    "quiet": run_num == 0,
                    "device": args.device,
                    "n_tests": args.n_tests,
                },
            )
            # Let some time for the child process to allocate the GPU
            time.sleep(args.delay)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
