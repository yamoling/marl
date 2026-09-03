"""Evaluate saved policy checkpoints on a run's train or test pool.

Example:
    uv run python scripts/test_policy_on_train_envs.py test logs/5x5_2agents_1laser-cooperative-dqn-1
"""

from __future__ import annotations

import argparse
import multiprocessing
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from lle import Action
from lle.characterization.plan import profile_plan

import marl
from marl import Agent
from marl.runners import compute_test_seed, seeded_rollout

EPISODES_PER_CHECKPOINT = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Device to evaluate on: 'gpu' (default, current behaviour) or 'cpu'.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Task:
    runpath: Path
    train_steps: list[int]
    test_steps: list[int]
    device: str = "gpu"

    def all_steps(self) -> list[int]:
        return list(set(self.train_steps + self.test_steps))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _evaluate_episode(env: Any, agent: Agent, checkpoint_step: int, test_num: int) -> dict[str, Any]:
    seed = compute_test_seed(checkpoint_step, test_num)
    episode, _, _ = seeded_rollout(env, agent, seed)
    plan = [[Action(int(action)) for action in np.array(joint_action)] for joint_action in episode.actions]
    world = env.unwrapped.current.world
    agent_status = {
        field: value
        for agent in world.agents
        for field, value in (
            (f"agent-{agent.num}-exited", agent.has_arrived),
            (f"agent-{agent.num}-alive", agent.is_alive),
        )
    }
    profile = profile_plan(world, plan)
    return {
        **episode.metrics,
        **agent_status,
        "cooperative-trajectory": profile.is_cooperative,
        "asymmetric-trajectory": profile.is_asymmetric,
        "chained-trajectory": profile.is_sequential(2),
        "convergent-trajectory": profile.is_convergent(),
        "divergent-trajectory": profile.is_divergent(),
        "interdependent-trajectory": profile.is_interdependent(2),
        "timestamp_sec": time.time(),
        "time_step": checkpoint_step,
        "test_num": test_num,
    }


def process_task(task: Task) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    train_logs: list[dict[str, Any]] = []
    test_logs: list[dict[str, Any]] = []
    try:
        run = marl.Run.load(task.runpath)
        if task.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(
                f"cuda:{run.seed % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
            )
        print(f"Processing {task.runpath} on {device}")
        agent = run.make_agent().to(device)
        train_env, test_env = None, None
        for time_step in task.all_steps():
            directory = run.get_saved_algo_dir(time_step)
            agent.load(directory)
            if time_step in task.train_steps:
                if train_env is None:
                    train_env = run.env.make()
                for test_num in range(EPISODES_PER_CHECKPOINT):
                    train_logs.append(_evaluate_episode(train_env, agent, time_step, test_num))
            if time_step in task.test_steps:
                if test_env is None:
                    test_env = run.test_env.make()
                for test_num in range(EPISODES_PER_CHECKPOINT):
                    test_logs.append(_evaluate_episode(test_env, agent, time_step, test_num))
    except Exception:  # noqa: BLE001 - preserve partial results from any evaluation failure
        return train_logs, test_logs, traceback.format_exc()
    return train_logs, test_logs, None


def discover_checkpoint_steps(runpath: Path) -> list[int]:
    """Return the sorted time steps stored in ``runpath/test`` checkpoint directories."""
    test_directory = runpath / "test"
    return sorted(int(path.name) for path in test_directory.glob("*") if path.is_dir() and path.name.isdigit())


def gather_missing_time_steps(filepath: Path, timesteps: list[int]) -> set[int]:
    try:
        df = pl.read_csv(filepath)
        missing = set(timesteps)
        for (t, *_), group in df.group_by("time_step"):
            if group.height == 500 and t in missing:
                missing.remove(t)
        return missing
    except FileNotFoundError:
        return set(timesteps)


def collect_tasks(
    logdir: Path,
    *,
    overwrite: bool,
    seed: int | None = None,
    checkpoint_steps: bool = False,
    device: str = "gpu",
):
    exp = marl.Experiment.load(logdir)
    if seed is not None:
        run = exp.get_run(seed)
        if run is None:
            raise ValueError(f"No run with seed {seed} found in experiment directory: {logdir}")
        runs = [run]
    else:
        runs = list(exp.runs)
    tasks = list[Task]()
    for run in runs:
        discovered_steps = discover_checkpoint_steps(run.runpath)
        timesteps = discovered_steps if checkpoint_steps else discovered_steps[-1:]
        if overwrite:
            missing_train = timesteps
            missing_test = timesteps
        else:
            # Check if the time steps are already present in both pools
            missing_train = gather_missing_time_steps(run.runpath / "test-policy-on-train-envs.csv", timesteps)
            missing_test = gather_missing_time_steps(run.runpath / "test-policy-on-test-envs.csv", timesteps)
            if len(missing_train) == 0 and len(missing_test) == 0:
                continue
        tasks.append(
            Task(runpath=run.runpath, train_steps=list(missing_train), test_steps=list(missing_test), device=device)
        )
    return tasks


def write_pool_results(filepath: Path, logs: list[dict[str, Any]], *, overwrite: bool) -> None:
    results = pl.DataFrame(logs)
    if not overwrite:
        try:
            existing = pl.read_csv(filepath)
            results = pl.concat([existing, results], how="diagonal_relaxed")
        except FileNotFoundError:
            pass
    results.write_csv(filepath)


def write_results(runpath: Path, train_logs: list[dict], test_logs: list[dict], *, overwrite: bool):
    if len(train_logs) > 0:
        write_pool_results(runpath / "test-policy-on-train-envs.csv", train_logs, overwrite=overwrite)
    if len(test_logs) > 0:
        write_pool_results(runpath / "test-policy-on-test-envs.csv", test_logs, overwrite=overwrite)


def process_logdirs(
    paths: list[Path],
    n_jobs: int,
    shuffle: bool,
    seed: int | None,
    overwrite: bool,
    checkpoint_steps: bool,
    dry_run: bool,
    device: str = "gpu",
):
    tasks = [
        t
        for logdir in paths
        for t in collect_tasks(logdir, overwrite=overwrite, seed=seed, checkpoint_steps=checkpoint_steps, device=device)
    ]
    print(f"Found {len(tasks)} tasks to run")
    if shuffle:
        random.shuffle(tasks)
    if dry_run:
        for t in tasks:
            train_episodes = len(t.train_steps) * EPISODES_PER_CHECKPOINT
            test_episodes = len(t.test_steps) * EPISODES_PER_CHECKPOINT
            print(f"  {t.runpath}\t{train_episodes} train steps {test_episodes} test steps to run")
        return
    if len(tasks) == 0:
        print("No task to run")
        return
    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(tasks)),
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                train, test, error = future.result()
            except Exception:  # noqa: BLE001 - one failed worker must not discard other workers' results
                print(f"[error]\t{task.runpath}: worker process failed", file=sys.stderr, flush=True)
                traceback.print_exc()
                continue

            try:
                write_results(task.runpath, train, test, overwrite=overwrite)
            except Exception:  # noqa: BLE001 - report the failure and continue saving other workers' results
                print(f"[error]\t{task.runpath}: could not save evaluation results", file=sys.stderr, flush=True)
                traceback.print_exc()
                continue

            status = "partial" if error is not None else "complete"
            print(f"[{status}]\t{task.runpath}: {len(train)} train | {len(test)} test")
            if error is not None:
                print(f"[error]\t{task.runpath}:\n{error}", file=sys.stderr, flush=True)


def main():
    args = parse_args()
    process_logdirs(
        args.logdirs,
        args.n_jobs,
        args.shuffle,
        args.seed,
        args.overwrite,
        args.checkpoint_steps,
        args.dry_run,
        args.device,
    )


if __name__ == "__main__":
    main()
