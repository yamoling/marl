import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import dotenv
import optuna
import typed_argparse as tap
from lle import World
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial
from tuning import Algo, make_trainer

import marl
from marl.env import LLEPool

N_STEPS = 1_000_000
POOL_SIZE = 500
N_TESTS = 500
TEST_OFFSET = 500
SETTING = "cooperative"
ALGOS: tuple[Algo, ...] = ("vdn", "qmix", "mappo", "dqn", "ippo")


@dataclass(frozen=True)
class PoolSpec:
    path: Path
    time_limit: int
    study_map_name: str

    @property
    def map_name(self):
        return self.path.name


class Args(tap.TypedArgs):
    pool_dir: Path = tap.arg(positional=True, help="Directory containing the pool of maps to train on.")
    n_seeds: int = tap.arg("--n-seeds", default=10)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_steps: int = tap.arg("--n-steps", default=N_STEPS)
    n_jobs: int = tap.arg("--n-jobs", default=8)
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    algos: list[Algo] = tap.arg(
        "--algos", default=list(ALGOS), nargs="+", help="Algorithms to train (defaults to all algorithms)."
    )
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    study_journal: Path = tap.arg("--study-journal", default=Path("optuna_study.journal"))
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False)
    skip_existing: bool = tap.arg("--skip-existing", default=True)
    test_interval: int = tap.arg("--test-interval", default=50_000)


def parse_pool_spec(pool_dir: Path):
    world = World.from_file(str(pool_dir / os.listdir(pool_dir)[0]))
    grid_size = world.width
    n_lasers = len(world.laser_sources)
    laser_label = "laser" if n_lasers == 1 else "lasers"
    study_map_name = f"{grid_size}x{grid_size}_agents{world.n_agents}_{laser_label}{n_lasers}"
    return PoolSpec(path=pool_dir, time_limit=grid_size**2, study_map_name=study_map_name)


def load_best_params(args: Args, algo: Algo, study_map_name: str):
    storage = JournalStorage(JournalFileBackend(args.study_journal.as_posix()))
    study_name = f"{algo.upper()}-{SETTING}-{study_map_name}"
    print(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError(f"Study {study_name!r} has no complete trials.")
    logging.info(f"Using best trial {study.best_trial.number} from {study_name} with value {study.best_value}")
    return study.best_params


def make_env(pool_dir: Path, pool_size: int, *, offset: int = 0, time_limit: int):
    return LLEPool(pool_dir, pool_size, offset=offset, time_limit=time_limit, state_type="flattened")


def experiment_logdir(spec: PoolSpec, algo: Algo):
    return Path("logs") / f"{spec.map_name}-{algo}"


def run_experiment(args: Args, spec: PoolSpec, algo: Algo):
    logdir = experiment_logdir(spec, algo)
    requested_seeds = range(args.start_seed, args.start_seed + args.n_seeds)
    if logdir.exists():
        exp = marl.Experiment.load(logdir)
        completed_seeds = {run.seed for run in exp.runs if run.is_complete and run.seed in requested_seeds}
        seeds = [seed for seed in requested_seeds if seed not in completed_seeds]
        if args.dry_run:
            logging.info(f"[exists] {len(seeds)} runs of {spec.map_name} / {algo} / pool={POOL_SIZE} -> {logdir}")
            return
        if len(seeds) == 0:
            if args.skip_existing:
                logging.info(
                    f"Skipping existing experiment: {logdir} ({len(completed_seeds)}/{args.n_seeds} runs complete)"
                )
                return
            raise FileExistsError(f"Experiment directory already exists: {logdir}")
        logging.info(
            f"Experiment {logdir} has only {len(completed_seeds)}/{args.n_seeds} complete runs; starting missing seeds {seeds}"
        )
    else:
        seeds = list(requested_seeds)
        if args.dry_run:
            logging.info(f"[new] {len(seeds)} runs of {spec.map_name} / {algo} / pool={POOL_SIZE} -> {logdir}")
            return
        train_env = make_env(spec.path, POOL_SIZE, time_limit=spec.time_limit)
        test_env = make_env(spec.path, N_TESTS, offset=TEST_OFFSET, time_limit=spec.time_limit)
        params = load_best_params(args, algo, spec.study_map_name)
        trainer = make_trainer(cast(optuna.Trial, FixedTrial(params)), algo, train_env)
        print(params)
        exp = marl.Experiment.create(train_env, trainer, test_env=test_env, logdir=logdir, n_steps=args.n_steps)
        logging.info("Created experiment in %s", exp.logdir)
    exp.run(
        seeds=seeds,
        save_weights=True,
        save_actions=True,
        test_interval=args.test_interval,
        n_tests=N_TESTS,
        n_jobs=args.n_jobs,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=args.quiet,
        limit_torch_threads=False,
    )


def main(args: Args):
    spec = parse_pool_spec(args.pool_dir)
    logging.info(f"Starting pool-500 sweep on {spec.map_name} with {len(args.algos)} algorithms.")

    for algo in args.algos:
        run_experiment(args, spec, algo)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("train_pool500.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.error(
            f"An error occurred while starting the pool-500 sweep with command line '{sys.argv}'.\nError: {e}",
            exc_info=True,
        )
        raise
