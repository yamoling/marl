import logging
import os
import subprocess
import sys
from pathlib import Path

import dotenv
import typed_argparse as tap

TRAIN_SCRIPT = Path(__file__).parent / "train_on_pool.py"
LOGDIR_PREFIX = "canonical-"
POOL_DIRS = (
    Path("layouts/canonicals/asymmetric"),
    Path("layouts/canonicals/convergent-2"),
    Path("layouts/canonicals/divergent-2"),
)


class Args(tap.TypedArgs):
    n_seeds: int = tap.arg("--n-seeds", default=16)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_jobs: int = tap.arg("--n-jobs", default=16)
    pool_size: int = tap.arg("--pool-size", default=500)
    n_tests: int = tap.arg("--n-tests", default=500)
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    test_interval: int = tap.arg("--test-interval", default=50_000)
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: str = tap.arg("--gpu-strategy", default="scatter")
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only print what would be trained.")


def build_command(args: Args, pool_dir: Path):
    """
    Build the command line invoking `train_on_pool.py` on the given pool.

    @ai-generated
    """
    command = [
        sys.executable,
        TRAIN_SCRIPT.as_posix(),
        pool_dir.as_posix(),
        "--n-seeds",
        str(args.n_seeds),
        "--start-seed",
        str(args.start_seed),
        "--n-jobs",
        str(args.n_jobs),
        "--pool-size",
        str(args.pool_size),
        "--n-tests",
        str(args.n_tests),
        "--n-steps",
        str(args.n_steps),
        "--test-interval",
        str(args.test_interval),
        "--gpu-strategy",
        args.gpu_strategy,
        "--logdir-prefix",
        LOGDIR_PREFIX,
    ]
    if len(args.disabled_gpus) > 0:
        command += ["--disabled-gpus", *(str(gpu) for gpu in args.disabled_gpus)]
    if args.dry_run:
        command.append("--dry-run")
    return command


def main(args: Args):
    """
    Sequentially train every canonical pool with all algorithms.

    @ai-generated
    """
    for pool_dir in POOL_DIRS:
        if not pool_dir.is_dir():
            raise FileNotFoundError(f"Pool directory not found: {pool_dir}")

    for pool_dir in POOL_DIRS:
        command = build_command(args, pool_dir)
        logging.info(f"Starting training on {pool_dir} with command: {' '.join(command)}")
        subprocess.run(command, check=True)
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
