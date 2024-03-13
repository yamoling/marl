from typing import Literal, Optional
import typed_argparse as tap

import os
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


def create_run(logdir: str, seed: int, n_tests: int, quiet: bool, device: Literal["auto", "cpu", "cuda"]):
    experiment = marl.Experiment.load(logdir)
    runner = experiment.create_runner(seed)
    runner.to(device)
    runner.train(n_tests=n_tests, quiet=quiet)


def main(args: Arguments):
    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    for i in range(args.n_runs - 1):
        seed = args.seed + i
        if os.fork() == 0:
            # Child processes are quiet if not specified otherwise
            if args.quiet is None:
                args.quiet = True
            args.seed = seed
            create_run(args.logdir, args.seed, args.n_tests, args.quiet, args.device)
            exit(0)

        # Sleep for some time for each child process to allocate GPUs properly
        time.sleep(args.delay)
    seed = args.seed + args.n_runs - 1
    if args.quiet is None:
        args.quiet = False
    create_run(args.logdir, seed, args.n_tests, args.quiet, args.device)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
