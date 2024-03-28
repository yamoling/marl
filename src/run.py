from typing import Literal, Optional
import typed_argparse as tap
from create_experiments import Args

import time
import marl


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=5)
    quiet: Optional[bool] = tap.arg(
        default=None, help="Run the experiment quietly. If 'None' and n_runs > 1, all runs are quiet except one."
    )
    delay: float = tap.arg(default=5.0, help="Delay in seconds between two consecutive runs")
    device: Literal["auto", "cpu", "cuda"] = tap.arg(default="auto")


def main(args: Arguments):
    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    experiment = marl.Experiment.load(args.logdir)
    for run_num in range(1, args.n_runs):
        seed = args.seed + run_num
        if args.quiet is None:
            args.quiet = True
        experiment.run(seed, args.quiet, args.n_tests, args.device, True)
        # Sleep for some time for each child process to allocate GPUs properly
        time.sleep(args.delay)
    if args.quiet is None:
        args.quiet = False
    experiment.run(args.seed, args.quiet, args.n_tests, args.device, False)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
