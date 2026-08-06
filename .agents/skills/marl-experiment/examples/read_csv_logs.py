"""Read local CSV logs lazily with Polars.

Run from the repository root after creating an experiment:
    uv run python .agents/skills/marl-experiment/examples/read_csv_logs.py logs/<experiment-name>
"""

import sys
from pathlib import Path

import polars as pl

from marl import Experiment


def main(logdir: Path) -> None:
    experiment = Experiment.load(logdir)

    frames: list[pl.LazyFrame] = []
    for run in experiment.runs:
        frames.append(run.test_metrics.select("time_step", "exit_rate").with_columns(seed=pl.lit(run.seed)))

    if not frames:
        print("No run directories found.")
        return

    per_run = pl.concat(frames).collect()
    aggregate = experiment.get_test_results(granularity=5_000).collect()
    print("Per-run test rows:")
    print(per_run)
    print("Cross-seed aggregate:")
    print(aggregate)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
