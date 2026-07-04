"""
Start a one or multiple new runs for an existing experiment.
"""

import logging
import os
import sys
from typing import Literal

import dotenv
import typed_argparse as tap

from marl.utils import DeviceLike


class Arguments(tap.TypedArgs):
    logdir: str = tap.arg(positional=True, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    n_jobs: int | None = tap.arg("--n-jobs", default=None, help="Maximal number of simultaneous processes to use")
    seed: int = tap.arg(default=0, help="The seed for the first run, subsequent ones are incremented by 1")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    test_interval: int = tap.arg(default=5_000)
    device: DeviceLike = tap.arg("--device", default="auto", help='Device to use ("auto", "cpu", "cuda:<gpu_id>", or int device index)')
    gpu_strategy: Literal["scatter", "group"] = tap.arg(default="group")
    render: bool = tap.arg(default=False, help="Render the tests")
    disabled_devices: list[int] = tap.arg(default=[], help="Disabled GPU devices", nargs="*")
    save_weights: bool = tap.arg("--save-weights", default=False, help="Whether to save the models' weights at each test interval")
    no_save_actions: bool = tap.arg("--no-save-actions", default=False, help="Disable saving test actions at each test interval.")
    quiet: bool = tap.arg(default=False, help="Quiet mode disables progressbars and console logs.")

    @property
    def seeds(self):
        return range(self.seed, self.seed + self.n_runs)


def main(args: Arguments):
    import marl

    experiment = marl.Experiment.load(args.logdir)
    if args.n_jobs is None:
        n_jobs = "auto"
    else:
        n_jobs = args.n_jobs
    print(args)
    experiment.run(
        args.seeds,
        gpu_strategy=args.gpu_strategy,
        save_weights=args.save_weights,
        save_actions=not args.no_save_actions,
        n_tests=args.n_tests,
        test_interval=args.test_interval,
        quiet=args.quiet,
        device=args.device,
        render_tests=args.render,
        n_jobs=n_jobs,
        disabled_gpus=args.disabled_devices,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("start_run.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Arguments).bind(main).run()
    except Exception as e:
        logging.error(f"An error occurred while starting a run with command line '{sys.argv}'.\nError: {e}", exc_info=True)
