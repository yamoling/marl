import time
import typed_argparse as tap

from typing import Literal, Optional
from torch.multiprocessing import Pool
from multiprocessing.pool import AsyncResult


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    _n_processes: Optional[int] = tap.arg("--n-processes", default=None, help="Maximal number of simultaneous processes to use")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    device: Literal["auto", "cpu"] | int = tap.arg(default="auto")
    estimated_memory_MB: int = tap.arg(default=3_000, help="Estimated memory in GB for the 'auto' device")
    gpu_strategy: Literal["fill", "conservative"] = tap.arg(default="conservative")

    @property
    def n_processes(self):
        if self._n_processes is not None:
            return min(self._n_processes, self.n_runs)

        # If we have GPUs, then start as many runs as there are GPUs
        import subprocess

        cmd = "nvidia-smi --list-gpus | wc -l"
        n_gpus = int(subprocess.check_output(cmd, shell=True).decode().strip())
        if n_gpus > 0:
            return min(n_gpus, self.n_runs)
        # Otherwise, start only one run at a time on the cpu
        return 1


def start_run(args: Arguments, run_num: int):
    import marl

    print(f"Starting run {run_num} with seed {args.seed + run_num}")
    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    experiment = marl.Experiment.load(args.logdir)
    experiment.run(
        seed=args.seed + run_num,
        fill_strategy=args.gpu_strategy,
        required_memory_MB=args.estimated_memory_MB,
        quiet=run_num > 0,
        device=args.device,
        n_tests=args.n_tests,
    )


def main(args: Arguments):
    with Pool(args.n_processes) as pool:
        handles = list[AsyncResult]()
        for run_num in range(args.n_runs):
            h = pool.apply_async(start_run, (args, run_num))
            handles.append(h)
            # If it is not the last process, wait a bit to let the time to allocate the GPUs correctly.
            if run_num != args.n_runs - 1:
                time.sleep(args.delay)

        for h in handles:
            h.get()


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
