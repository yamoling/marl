from typing import Literal, Optional
import typed_argparse as tap

import time

from marl.utils.others import DeviceStr


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    quiet: Optional[bool] = tap.arg(default=None, help="Run the experiment quietly. If 'None' all runs are quiet the first one.")
    delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    device: DeviceStr = tap.arg(default="auto")
    estimated_memory_MB: int = tap.arg(default=0, help="Estimated memory in GB for the 'auto' device")
    gpu_strategy: Literal["fill", "conservative"] = tap.arg(default="conservative")


def main(args: Arguments):
    # Import in the function to quicken the startup time
    import marl

    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    experiment = marl.Experiment.load(args.logdir)
    for run_num in range(0, args.n_runs - 1):
        seed = args.seed + run_num
        if args.quiet is None:
            args.quiet = True
        experiment.run(seed, args.quiet, args.n_tests, args.device, args.gpu_strategy, args.estimated_memory_MB, True)
        # Sleep for some time for each child process to allocate GPUs properly
        time.sleep(args.delay)
    if args.quiet is None:
        args.quiet = False
    experiment.run(args.seed + args.n_runs - 1, args.quiet, args.n_tests, args.device, args.gpu_strategy, args.estimated_memory_MB, False)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
