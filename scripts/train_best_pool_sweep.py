import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import dotenv
import optuna
import typed_argparse as tap
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial
from tuning import Algo, Setting, make_trainer

import marl
from marl.env import LLEPool

N_STEPS = 1_000_000
N_TESTS = 500
TEST_OFFSET = 500
POOL_SIZES = [1, 10, 20, 50, 100, 150, 200, 300, 400, 500]
ALGOS = ("vdn", "qmix", "mappo", "dqn", "ippo")
POOL_DIRS = (
    Path("maps/train/5x5_2agents_1laser/independent"),
    Path("maps/train/5x5_2agents_1laser/cooperative"),
    Path("maps/train/9x9_3agents_2lasers/independent"),
    Path("maps/train/9x9_3agents_2lasers/cooperative"),
)
DEFAULT_STUDY_MAP_NAME = "9x9_agents3_lasers2"


@dataclass(frozen=True)
class PoolSpec:
    path: Path
    setting: Setting
    time_limit: int

    @property
    def map_name(self):
        return self.path.parent.name

    @property
    def label(self):
        return f"{self.map_name}-{self.setting}"


class Args(tap.TypedArgs):
    n_seeds: int = tap.arg("--n-seeds", default=10)
    n_steps: int = tap.arg("--n-steps", default=N_STEPS)
    n_jobs: int = tap.arg("--n-jobs", default=8)
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    study_journal: Path = tap.arg("--study-journal", default=Path("optuna_study.journal"))
    study_map_name: str = tap.arg("--study-map-name", default=DEFAULT_STUDY_MAP_NAME)
    log_root: Path = tap.arg("--log-root", default=Path("logs/best-pool-sweep"))
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False)
    save_weights: bool = tap.arg("--save-weights", default=False)
    no_save_actions: bool = tap.arg("--no-save-actions", default=False)
    skip_existing: bool = tap.arg("--skip-existing", default=True)


def parse_pool_spec(pool_dir: Path) -> PoolSpec:
    setting_name = pool_dir.name
    if setting_name not in ("cooperative", "independent"):
        raise ValueError(f"Could not infer setting from pool directory: {pool_dir}")
    setting = cast(Setting, setting_name)

    match = re.match(r"(?P<size>\d+)x(?P=size)_", pool_dir.parent.name)
    if match is None:
        raise ValueError(f"Could not infer grid size from pool directory: {pool_dir}")

    grid_size = int(match.group("size"))
    return PoolSpec(path=pool_dir, setting=setting, time_limit=grid_size**2)


def load_best_params(args: Args, algo: Algo, setting: Setting):
    storage = JournalStorage(JournalFileBackend(args.study_journal.as_posix()))
    study_name = f"{algo.upper()}-{setting}-{args.study_map_name}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError(f"Study {study_name!r} has no complete trials.")
    logging.info("Using best trial %s from %s with value %s", study.best_trial.number, study_name, study.best_value)
    return study.best_params


def make_env(pool_dir: Path, pool_size: int, *, offset: int = 0, time_limit: int):
    return LLEPool(pool_dir, pool_size, offset=offset, time_limit=time_limit, state_type="layered")


def experiment_logdir(args: Args, spec: PoolSpec, algo: Algo, pool_size: int):
    return args.log_root / spec.map_name / spec.setting / algo / f"pool-{pool_size}"


def run_experiment(args: Args, spec: PoolSpec, algo: Algo, pool_size: int, best_params: dict):
    logdir = experiment_logdir(args, spec, algo, pool_size)
    if logdir.exists():
        if args.skip_existing:
            logging.info("Skipping existing experiment: %s", logdir)
            return
        raise FileExistsError(f"Experiment directory already exists: {logdir}")

    train_env = make_env(spec.path, pool_size, time_limit=spec.time_limit)
    test_env = make_env(spec.path, N_TESTS, offset=TEST_OFFSET, time_limit=spec.time_limit)
    trainer = make_trainer(cast(optuna.Trial, FixedTrial(best_params)), algo, train_env)

    if args.dry_run:
        logging.info("Would run %s / %s / pool=%s -> %s", spec.label, algo, pool_size, logdir)
        return

    exp = marl.Experiment.create(train_env, trainer, test_env=test_env, logdir=logdir, n_steps=args.n_steps)
    logging.info("Created experiment in %s", exp.logdir)
    exp.run(
        seeds=args.n_seeds,
        save_weights=args.save_weights,
        save_actions=not args.no_save_actions,
        test_interval=args.n_steps,
        n_tests=N_TESTS,
        n_jobs=args.n_jobs,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=args.quiet,
    )


def main(args: Args):
    specs = [parse_pool_spec(pool_dir) for pool_dir in POOL_DIRS]
    best_params = {
        (algo, setting): load_best_params(args, algo, setting)
        for setting in ("cooperative", "independent")
        for algo in ALGOS
    }

    n_total = len(specs) * len(ALGOS) * len(POOL_SIZES)
    logging.info("Starting best-parameter pool sweep with %s experiments.", n_total)
    for spec in specs:
        for algo in ALGOS:
            params = best_params[(algo, spec.setting)]
            for pool_size in POOL_SIZES:
                run_experiment(args, spec, algo, pool_size, params)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("train_best_pool_sweep.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.error(
            f"An error occurred while starting the pool sweep with command line '{sys.argv}'.\nError: {e}",
            exc_info=True,
        )
        raise
