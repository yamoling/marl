"""Evaluate saved policy checkpoints on a run's train or test pool.

Example:
    uv run python scripts/test_policy_on_train_envs.py test logs/5x5_2agents_1laser-cooperative-dqn-1
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import random
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    parser.add_argument("pool", choices=("train", "test", "both"), help="Environment pool to evaluate.")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluate only the run with this seed (default: evaluate every run).",
    )
    parser.add_argument(
        "--checkpoint-steps",
        action="store_true",
        help="Evaluate every saved checkpoint instead of only the latest one.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the order of the runs before evaluating.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def saved_checkpoints(run: Run) -> list[tuple[int, Path]]:
    """Return all numbered checkpoints that contain saved agent state."""
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
    return sorted(checkpoints)


def latest_checkpoint(run: Run) -> tuple[int, Path]:
    return saved_checkpoints(run)[-1]


def iter_runs(logdir: Path):
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
    agent_status = {
        field: value
        for agent in world.agents
        for field, value in (
            (f"agent-{agent.num}-exited", agent.has_arrived),
            (f"agent-{agent.num}-alive", agent.is_alive),
        )
    }
    return {
        **episode.metrics,
        **agent_status,
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


def append_rows(output: Path, rows: list[dict[str, object]]) -> None:
    """Append a batch of results, writing it durably before returning.

    CSV headers cannot be extended in place. If a later batch introduces a new
    field (for example, when runs have different numbers of agents), rewrite
    the existing rows with the expanded header before appending the batch.
    """
    if not rows:
        raise RuntimeError(f"Evaluation produced no episodes for {output.parent}")

    existing_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if output.exists() and output.stat().st_size:
        with output.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
    original_fieldnames = set(fieldnames)

    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

    needs_rewrite = any(fieldname not in original_fieldnames for fieldname in fieldnames)
    if needs_rewrite:
        with output.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(rows)
            csv_file.flush()
            os.fsync(csv_file.fileno())
        return

    mode = "a" if output.exists() and output.stat().st_size else "w"
    with output.open(mode, newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def process_logdir(
    logdir: Path,
    pool: Pool,
    *,
    dry_run: bool,
    overwrite: bool,
    n_jobs: int,
    seed: int | None = None,
    checkpoint_steps: bool = False,
    shuffle: bool = False,
) -> int:
    runs = iter_runs(logdir)
    if seed is not None:
        runs = [run for run in runs if run.seed == seed]
        if not runs:
            raise ValueError(f"No run with seed {seed} found in experiment directory: {logdir}")

    pending: list[tuple[Run, Path, int, Path]] = []
    for run in runs:
        try:
            checkpoints = saved_checkpoints(run) if checkpoint_steps else [latest_checkpoint(run)]
        except FileNotFoundError as error:
            print(f"[skip no checkpoints] {error}", flush=True)
            continue

        for current_step, checkpoint_dir in checkpoints:
            output = run.runpath / f"test-policy-on-{pool}-envs.csv"
            description = f"{run.runpath}: checkpoint={current_step}, {pool}-pool episodes={pool_size(run, pool)}, output={output}"

            if output_has_data(output) and not overwrite:
                print(f"[skip existing] {description}", flush=True)
                continue
            if dry_run:
                print(f"[dry run] {description}", flush=True)
                continue
            pending.append((run, output, current_step, checkpoint_dir))

    if dry_run or not pending:
        return 0
    if shuffle:
        random.shuffle(pending)

    # Start fresh once per output when requested. Subsequent checkpoint results
    # are appended as workers finish, so completed work survives interruptions.
    outputs = {output for _run, output, _step, _checkpoint_dir in pending}
    if overwrite:
        for output in outputs:
            output.unlink(missing_ok=True)

    rows_by_output: dict[Path, int] = {output: 0 for output in outputs}
    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(pending)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(evaluate_run, (run.runpath, checkpoint_step, checkpoint_dir, pool)): (
                run,
                output,
                checkpoint_step,
            )
            for run, output, checkpoint_step, checkpoint_dir in pending
        }
        for future in as_completed(futures):
            run, output, checkpoint_step = futures[future]
            rows = future.result()
            append_rows(output, rows)
            rows_by_output[output] += len(rows)
            print(
                f"[completed] {run.runpath}: checkpoint={checkpoint_step}, "
                f"{pool}-pool episodes={len(rows)} ({rows_by_output[output]} accumulated), "
                f"saved to {output}",
                flush=True,
            )

    return len(outputs)


def main() -> int:
    args = parse_args()
    pools: tuple[Pool, ...] = ("test", "train") if args.pool == "both" else (cast(Pool, args.pool),)
    failures = 0
    for pool in pools:
        for logdir in args.logdirs:
            try:
                process_logdir(
                    logdir,
                    pool,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    n_jobs=args.n_jobs,
                    seed=args.seed,
                    checkpoint_steps=args.checkpoint_steps,
                    shuffle=args.shuffle,
                )
            except Exception as error:
                failures += 1
                print(f"[failed] {logdir} ({pool}): {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
