"""Evaluate each run's latest test-policy checkpoint on its train or test pool.

Example:
    uv run python scripts/test_policy_on_train_envs.py test logs/5x5_2agents_1laser-cooperative-dqn-1
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal, cast

import torch
from lle import Action
from lle.characterization.plan import PlanProfile, profile_plan

import marl
from marl import Agent
from marl.models.run import Run
from marl.runners import compute_test_seed, seeded_rollout

Pool = Literal["train", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", choices=("train", "test"), help="Environment pool to evaluate.")
    parser.add_argument("logdirs", type=Path, nargs="+", help="Experiment log directories to evaluate.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned evaluations without loading checkpoints or writing CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV file instead of skipping that run.",
    )
    parser.add_argument(
        "--n-jobs",
        type=positive_int,
        default=1,
        help="Number of runs to evaluate in parallel (default: 1).",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def latest_checkpoint(run: Run) -> tuple[int, Path]:
    """Return the highest numbered checkpoint that contains saved agent state."""
    test_root = run.runpath / "test"
    checkpoints = (
        [
            (int(path.name), path)
            for path in test_root.iterdir()
            if path.is_dir() and path.name.isdigit() and any(path.iterdir())
        ]
        if test_root.exists()
        else []
    )
    if not checkpoints:
        raise FileNotFoundError(f"No saved checkpoints found in {test_root}")
    return max(checkpoints, key=lambda checkpoint: checkpoint[0])


def iter_runs(logdir: Path) -> Iterable[Run]:
    if not marl.Experiment.is_experiment_directory(logdir):
        raise ValueError(f"Not an experiment directory (missing experiment.json): {logdir}")

    experiment = marl.Experiment.load(logdir)
    runs = sorted((run.to_full() for run in experiment.runs), key=lambda run: run.seed)
    if not runs:
        raise ValueError(f"No runs found in experiment directory: {logdir}")
    return runs


def pool_config(run: Run, pool: Pool) -> Any:
    return run.env if pool == "train" else run.test_env


def pool_size(run: Run, pool: Pool) -> int:
    return int(pool_config(run, pool).size)


def evaluate_run(task: tuple[Path, int, Path, Pool]) -> list[dict[str, object]]:
    """Evaluate one run's saved policy on every environment in the selected pool.

    This function is executed in a run-level worker. The worker reconstructs the
    agent and environment from disk, then evaluates all episodes for the run
    serially so model and environment state stay isolated to that process.
    """
    rundir, checkpoint_step, checkpoint_dir, pool = task
    run = Run.load(rundir)
    device = torch.device(f"cuda:{run.seed % 3}") if torch.cuda.is_available() else torch.device("cpu")
    agent = run.make_agent().to(device)
    agent.load(checkpoint_dir)
    env = pool_config(run, pool).make()
    return [_evaluate_episode(env, agent, checkpoint_step, test_num) for test_num in range(pool_size(run, pool))]


def _evaluate_episode(env: Any, agent: Agent, checkpoint_step: int, test_num: int) -> dict[str, object]:
    seed = compute_test_seed(checkpoint_step, test_num)
    episode, _, _ = seeded_rollout(env, agent, seed)
    plan = [[Action(int(action)) for action in cast(Iterable[Any], joint_action)] for joint_action in episode.actions]
    world = env.unwrapped.current.world
    profile: PlanProfile = profile_plan(world, plan)
    exit_status = {f"agent-{agent.num}-exited": agent.has_arrived for agent in world.agents}
    return {
        **episode.metrics,
        **exit_status,
        "cooperative-trajectory": profile.is_cooperative,
        "asymmetric-trajectory": profile.is_asymmetric,
        "chained-trajectory": profile.is_chained(2),
        "convergent-trajectory": profile.is_convergent(),
        "divergent-trajectory": profile.is_divergent(),
        "interdependent-trajectory": profile.is_interdependent(2),
        "timestamp_sec": time.time(),
        "time_step": checkpoint_step,
    }


def output_has_data(output: Path) -> bool:
    """Return whether an existing output has a header and at least one result row."""
    try:
        with output.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Header
            return next(reader, None) is not None
    except (FileNotFoundError, StopIteration):
        return False


def write_rows(output: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Evaluation produced no episodes for {output.parent}")

    fieldnames = list(rows[0])
    for row in rows[1:]:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_logdir(logdir: Path, pool: Pool, *, dry_run: bool, overwrite: bool, n_jobs: int) -> int:
    runs = iter_runs(logdir)
    pending: list[tuple[Run, Path, int, Path]] = []
    for run in runs:
        output = run.runpath / f"test-policy-on-{pool}-envs.csv"
        checkpoint_step, checkpoint_dir = latest_checkpoint(run)
        description = (
            f"{run.runpath}: checkpoint={checkpoint_step}, {pool}-pool episodes={pool_size(run, pool)}, output={output}"
        )

        if output_has_data(output) and not overwrite:
            print(f"[skip existing] {description}")
            continue
        if dry_run:
            print(f"[dry run] {description}")
            continue
        pending.append((run, output, checkpoint_step, checkpoint_dir))

    if dry_run or not pending:
        return 0

    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(pending)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        jobs = (
            (run.runpath, checkpoint_step, checkpoint_dir, pool)
            for run, _output, checkpoint_step, checkpoint_dir in pending
        )
        results = executor.map(evaluate_run, jobs)
        for (run, output, checkpoint_step, checkpoint_dir), rows in zip(pending, results, strict=True):
            description = (
                f"{run.runpath}: checkpoint={checkpoint_step}, {pool}-pool episodes={pool_size(run, pool)}, "
                f"output={output}"
            )
            write_rows(output, rows)
            print(f"[written] {description}")

    return len(pending)


def main() -> int:
    args = parse_args()
    pool = cast(Pool, args.pool)
    failures = 0
    for logdir in args.logdirs:
        try:
            process_logdir(logdir, pool, dry_run=args.dry_run, overwrite=args.overwrite, n_jobs=args.n_jobs)
        except Exception as error:
            failures += 1
            print(f"[failed] {logdir}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
