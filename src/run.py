from typing import Literal
import typed_argparse as tap
from create_experiments import Args

import os
import time
import marl


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=5)
    quiet: bool = tap.arg(default=False)
    delay: float = tap.arg(default=1.0, help="Delay in seconds between two consecutive runs")
    device: Literal["auto", "cpu", "cuda"] = tap.arg(default="auto")


def create_run(args: Arguments):
    experiment = marl.Experiment.load(args.logdir)
    runner = experiment.create_runner(seed=args.seed)
    runner.to(args.device)
    runner.train(n_tests=args.n_tests)


def main(args: Arguments):
    # Load the experiment from disk and start a child process for each run.
    # The run with seed=0 is spawned in the main process.
    for i in range(args.n_runs - 1):
        seed = args.seed + i
        if os.fork() == 0:
            # Force child processes to be quiet
            args.quiet = True
            args.seed = seed
            create_run(args)
            exit(0)

        # Sleep for some time for each child process to allocate GPUs properly
        time.sleep(args.delay)
    seed = args.seed + args.n_runs - 1
    create_run(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
