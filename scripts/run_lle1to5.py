"""
Run MAPPO, IPPO, DQN (independent, no mixer), VDN and QMIX on LLE levels 1 to 5 (included) with
layered observations and 10 seeds. The number of time steps depends on the level (see `N_STEPS`).

The hyperparameters of every algorithm are those of the best trial of its study in the LLE6 tuning
journal (`tunings/lle6-layered-500k.journal`, studies named `<name>-<algo>-score-0`). The trainers are
rebuilt by replaying these parameters with a `FixedTrial` into the search space that produced the
journal (`make_trainer` of `scripts/tune_lle6.py`). Studies that are not in the journal yet
(e.g. DQN and IPPO) raise an error when their experiment has to be created.

All runs of all experiments share a single job queue. Existing experiments are reused and only
missing or incomplete seeds are launched. Incomplete runs restart from step zero, replacing their
partial metrics; completed and actively running seeds are left untouched.

With `--sample-at-test`, the PPO agents (MAPPO and IPPO) sample from their policy at test time instead
of taking the argmax. These experiments are stored in `logs/lle<level>-<algo>-sampled`.

Run with `uv run python scripts/run_lle1to5.py [options]` from the repository root, e.g.
`uv run python scripts/run_lle1to5.py --levels 1 2 3 --algos mappo ippo --sample-at-test`.

@ai-generated
"""

import functools
import logging
from pathlib import Path
from typing import Literal, cast

import optuna
import torch
import typed_argparse as tap
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial
from tune_lle6 import make_trainer as suggest_trainer

from marl import Experiment
from marl.algos import PPO
from marl.env import LLEConfig
from marl.models import Run, Trainer
from marl.runners import parallel_run
from marl.utils import DeviceLike

logger = logging.getLogger(__name__)

LEVELS = [1, 2, 3, 4, 5, 6]
Algo = Literal["mappo", "ippo", "dqn", "vdn", "qmix"]
ALGOS: tuple[Algo, ...] = ("mappo", "ippo", "dqn", "vdn", "qmix")
N_STEPS = {1: 400_000, 2: 400_000, 3: 600_000, 4: 600_000, 5: 1_000_000, 6: 1_000_000}
"""Number of training time steps per LLE level."""
DEFAULT_JOURNAL = Path("tunings", "lle6-layered-500k.journal")
DEFAULT_STUDY_NAME = "lle6-layered-500k"
"""Search name of the studies; their full name is `<name>-<algo>-score-0`."""


@functools.cache
def load_best_params(journal: Path, study_name: str, algo: Algo) -> dict:
    """Read the best hyperparameters of an algorithm from an Optuna journal. @ai-generated"""
    if not journal.exists():
        raise FileNotFoundError(f"No tuning journal at {journal}.")
    storage = JournalStorage(JournalFileBackend(journal.as_posix()))
    full_name = f"{study_name}-{algo}-score-0"
    try:
        study = optuna.load_study(study_name=full_name, storage=storage)
    except KeyError:
        available = sorted(summary.study_name for summary in optuna.get_all_study_summaries(storage))
        raise KeyError(f"{journal} has no study {full_name!r}. Available studies: {available}") from None
    n_complete = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if n_complete == 0:
        raise RuntimeError(f"Study {full_name!r} has no complete trials.")
    logger.info(f"Using best trial {study.best_trial.number} of {full_name} (value {study.best_value}, {n_complete} complete trials)")
    return study.best_params


def make_trainer(env: LLEConfig, algo: Algo, params: dict, sample_at_test: bool = False) -> Trainer:
    """Rebuild the trainer of `algo` from the tuned parameters of the journal. @ai-generated"""
    trainer = suggest_trainer(cast(optuna.Trial, FixedTrial(params)), algo, env)
    if sample_at_test:
        assert isinstance(trainer, PPO)
        trainer.greedy_at_test = False
    return trainer


class Args(tap.TypedArgs):
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only print what would be created/run.")
    no_training_data: bool = tap.arg(
        "--no-training-data", default=False, help="Do not log trainer-update metrics; retain training-episode and test metrics."
    )
    levels: list[int] = tap.arg("--levels", default=list(LEVELS), nargs="*", help=f"LLE levels to run (default: all, {list(LEVELS)}).")
    start_seed: int = tap.arg("--start-seed", default=0, help="First seed to run (inclusive).")
    n_seeds: int = tap.arg("--n-seeds", default=10, help="Number of seeds per experiment (start_seed to start_seed + n_seeds, exclusive).")
    device: DeviceLike = tap.arg("--device", default="auto", help='"auto", "cpu", "cuda:<id>" or a GPU index.')
    n_jobs: int | None = tap.arg("--n-jobs", default=None, help="Number of parallel run processes (defaults to the number of usable GPUs).")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    test_interval: int = tap.arg("--test-interval", default=10_000)
    n_tests: int = tap.arg("--n-tests", default=5)
    journal: Path = tap.arg("--journal", default=DEFAULT_JOURNAL, help="Optuna journal providing the hyperparameters.")
    study_name: str = tap.arg("--study-name", default=DEFAULT_STUDY_NAME, help="Search name of the studies to read.")
    algos: list[Algo] = tap.arg("--algos", default=list(ALGOS), nargs="*", help=f"Algorithms to run (default: all, {list(ALGOS)}).")
    sample_at_test: bool = tap.arg(
        "--sample-at-test", default=False, help="PPO agents sample at test time instead of argmax (logdir suffix '-sampled')."
    )


def collect_runs(args: Args) -> list[Run]:
    """
    Create missing experiments and queue missing or incomplete seeds, excluding active runs.
        Incomplete seeds restart from step zero rather than resuming from a checkpoint.

    @ai-edited
    """
    invalid = set(args.levels) - set(LEVELS)
    if invalid:
        raise ValueError(f"Invalid level(s) {sorted(invalid)}; available levels are {list(LEVELS)}.")
    invalid_algos = set(args.algos) - set(ALGOS)
    if invalid_algos:
        raise ValueError(f"Invalid algorithm(s) {sorted(invalid_algos)}; available algorithms are {list(ALGOS)}.")
    if args.sample_at_test and not set(args.algos) <= {"mappo", "ippo"}:
        raise ValueError("--sample-at-test only applies to PPO algorithms: use it with `--algos mappo ippo`.")
    suffix = "-sampled" if args.sample_at_test else ""
    runs = []
    for level in cast(list[Literal[1, 2, 3, 4, 5, 6]], args.levels):
        for algo in args.algos:
            name = f"lle{level}-{algo}{suffix}"
            logdir = Path("logs") / name
            if logdir.exists():
                exp = Experiment.load(logdir)
                existing = {run.seed for run in exp.runs if run.is_running or run.is_complete}
            else:
                exp = None
                existing = set()
            seeds = [s for s in range(args.start_seed, args.start_seed + args.n_seeds) if s not in existing]
            if not seeds:
                logger.info(f"[skip] {name}: all seeds are complete or currently running.")
                continue
            if args.dry_run:
                if exp is None:
                    load_best_params(args.journal, args.study_name, algo)
                logger.info(f"[dry-run] {name}: would run seeds {seeds} -> {logdir}")
                continue
            if exp is None:
                env = LLEConfig(level, obs_type="layered", state_type="flattened")
                params = load_best_params(args.journal, args.study_name, algo)
                exp = Experiment.create(env, make_trainer(env, algo, params, args.sample_at_test), logdir=name, n_steps=N_STEPS[level])
                logger.info(f"[new] Created experiment {exp.logdir} with parameters {params}")
            logger.info(f"{name}: queueing seeds {seeds}")
            runs += exp.create_runs(
                seeds,
                args.n_tests,
                args.test_interval,
                save_weights=False,
                save_actions=False,
                save_training_data=not args.no_training_data,
            )
    return runs


def main(args: Args):
    runs = collect_runs(args)
    if not runs:
        return
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = 1 if args.device == "cpu" else max(1, torch.cuda.device_count() - len(args.disabled_gpus))
    logger.info(f"Launching {len(runs)} run(s) with {n_jobs} parallel job(s).")
    parallel_run(
        runs,
        n_jobs=n_jobs,
        device=args.device,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tap.Parser(Args).bind(main).run()
