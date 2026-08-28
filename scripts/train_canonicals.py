"""
Train all algorithms on the canonical pools (asymmetric, convergent-2, divergent-2).

Each pool is trained with 16 seeds and 16 parallel workers on a pool of 500 maps,
using the best hyperparameters found by the cooperative tuning studies.

Usage:
    python scripts/train_canonicals.py [--dry-run] [--n-seeds 16] ...
"""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import dotenv
import train_on_pool
import typed_argparse as tap
from tuning import Algo

LOGDIR_PREFIX = "canonical-"
POOL_DIRS = (
    Path("layouts/canonicals/asymmetric"),
    Path("layouts/canonicals/convergent-2"),
    Path("layouts/canonicals/divergent-2"),
    Path("layouts/canonicals/interdependent-2"),
    Path("layouts/canonicals/sequential-2"),
)


class Args(tap.TypedArgs):
    n_seeds: int = tap.arg("--n-seeds", default=16)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_jobs: int = tap.arg("--n-jobs", default=16)
    pool_size: int = tap.arg("--pool-size", default=500)
    n_tests: int = tap.arg("--n-tests", default=500)
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    test_interval: int = tap.arg("--test-interval", default=50_000)
    algos: list[Algo] = tap.arg("--algos", default=list(train_on_pool.ALGOS), nargs="+")
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    study_journal: Path = tap.arg("--study-journal", default=Path("optuna_study.journal"))
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only print what would be trained.")
    skip_existing: bool = tap.arg("--skip-existing", default=True)


def make_pool_args(args: Args, pool_dir: Path):
    """
    Build the `train_on_pool` arguments for the given canonical pool.

    @ai-generated
    """
    return train_on_pool.Args(
        pool_dir=pool_dir,
        n_seeds=args.n_seeds,
        start_seed=args.start_seed,
        n_steps=args.n_steps,
        n_jobs=args.n_jobs,
        pool_size=args.pool_size,
        n_tests=args.n_tests,
        disabled_gpus=args.disabled_gpus,
        algos=args.algos,
        gpu_strategy=args.gpu_strategy,
        study_journal=args.study_journal,
        quiet=args.quiet,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        test_interval=args.test_interval,
        logdir_prefix=LOGDIR_PREFIX,
    )


def main(args: Args):
    """
    Sequentially train every canonical pool with all the requested algorithms.

    @ai-generated
    """
    for pool_dir in POOL_DIRS:
        if not pool_dir.is_dir():
            raise FileNotFoundError(f"Pool directory not found: {pool_dir}")

    for pool_dir in POOL_DIRS:
        logging.info(f"Starting training on {pool_dir} with algorithms {args.algos}")
        train_on_pool.main(make_pool_args(args, pool_dir))
        logging.info(f"Finished training on {pool_dir}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("train_canonicals.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.error(
            f"An error occurred while starting the canonical sweep with '{sys.argv}'.\nError: {e}", exc_info=True
        )
        raise
