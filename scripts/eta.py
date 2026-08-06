"""Compute training throughput, progress, and ETA across experiment runs.

Example:
    python scripts/steps_per_second.py logs/sequential-2-dqn
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

TIME_STEP = "time_step"
TIMESTAMP = "timestamp_sec"


@dataclass(frozen=True)
class RunProgress:
    path: Path
    step: float
    steps_per_second: float | None
    eta_seconds: float | None
    elapsed_seconds: float


def read_run_progress(train_file: Path, total_steps: float) -> RunProgress:
    """Read progress from ``train.csv`` and the latest step from ``test.csv``."""
    with train_file.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        if rows.fieldnames is None or TIME_STEP not in rows.fieldnames or TIMESTAMP not in rows.fieldnames:
            raise ValueError(f"{train_file} must contain {TIME_STEP!r} and {TIMESTAMP!r} columns")

        first_timestamp: float | None = None
        last_timestamp: float | None = None
        last_step: float | None = None
        for row in rows:
            try:
                timestamp = float(row[TIMESTAMP])
                step = float(row[TIME_STEP])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{train_file} contains a non-numeric step or timestamp") from error
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            last_step = step

    if first_timestamp is None or last_timestamp is None or last_step is None:
        raise ValueError(f"{train_file} is empty")

    latest_step = last_step
    test_file = train_file.with_name("test.csv")
    latest_test_step = 0.0
    if test_file.is_file():
        with test_file.open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            if rows.fieldnames is None or TIME_STEP not in rows.fieldnames:
                pass
            else:
                for row in rows:
                    try:
                        latest_test_step = float(row[TIME_STEP])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{test_file} contains a non-numeric step") from error
        latest_step = max(latest_step, latest_test_step)

    elapsed = last_timestamp - first_timestamp
    rate = last_step / elapsed if elapsed > 0 else None
    remaining_steps = max(0.0, total_steps - latest_step)
    eta = remaining_steps / rate if rate is not None and rate > 0 else None
    return RunProgress(train_file.parent, latest_step, rate, eta, elapsed)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = round(seconds)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def run_steps_per_second(train_file: Path) -> float:
    """Return the steps-per-second rate recorded in one run's ``train.csv``."""
    progress = read_run_progress(train_file, total_steps=0)
    if progress.steps_per_second is None:
        raise ValueError(f"{train_file} has a non-positive elapsed time")
    return progress.steps_per_second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="experiment directory containing run-* subdirectories")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    experiment_path = experiment_dir / "experiment.json"
    try:
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        total_steps = float(experiment["n_steps"])
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing experiment metadata: {experiment_path}") from error
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{experiment_path} must contain a numeric n_steps value") from error

    train_files = sorted(experiment_dir.glob("run-*/train.csv"))
    if not train_files:
        raise FileNotFoundError(f"No run-*/train.csv files found in {experiment_dir}")

    progress: list[RunProgress] = []
    for train_file in train_files:
        try:
            progress.append(read_run_progress(train_file, total_steps))
        except ValueError as error:
            print(f"Skipping {train_file}: {error}")

    if not progress:
        raise ValueError(f"No valid train.csv files found in {experiment_dir}")

    print(f"Target steps: {total_steps:g}")
    print("Current progress:")
    for run in progress:
        percentage = min(100.0, 100.0 * run.step / total_steps) if total_steps > 0 else 100.0
        status = (
            f"complete in {format_duration(run.elapsed_seconds)}"
            if run.step >= total_steps
            else f"ETA {format_duration(run.eta_seconds)}"
        )
        rate = "unknown" if run.steps_per_second is None else f"{run.steps_per_second:.6f} steps/s"
        print(f"  {run.path.name}: {run.step:g}/{total_steps:g} steps ({percentage:.2f}%) - {rate} - {status}")

    rates = [run.steps_per_second for run in progress if run.steps_per_second is not None]
    if not rates:
        raise ValueError(f"No valid throughput measurements found in {experiment_dir}")
    print(f"Average steps per second: {fmean(rates):.6f} ({len(rates)} runs)")

    unfinished_etas = [run.eta_seconds for run in progress if run.step < total_steps and run.eta_seconds is not None]
    unknown_eta = any(run.step < total_steps and run.eta_seconds is None for run in progress)
    overall_eta = None if unknown_eta else max(unfinished_etas, default=0.0)
    print(f"ETA for all runs: {format_duration(overall_eta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
