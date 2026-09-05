"""Export per-map median and interquartile win rates from LAN experiment CSV logs."""

import argparse
from pathlib import Path

import polars as pl

from marl import Experiment


def main():
    """Aggregate seeds using the paper's median and quartiles, retaining sample counts. @ai-generated"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="For example logs/lan-paper-v3/lan")
    parser.add_argument("--metric", default="battle_won")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for spec in sorted(args.root.glob("*/experiment.json")):
        experiment = Experiment.load(spec.parent)
        for run in experiment.runs:
            if not run.is_complete:
                print(f"Skipping incomplete run {run.rundir}")
                continue
            # Each row already averages the 32 evaluation episodes for this seed.
            frames.append(
                run.test_metrics.select("time_step", args.metric).with_columns(
                    pl.lit(spec.parent.name).alias("map"), pl.lit(run.seed).alias("seed")
                )
            )
    if not frames:
        raise ValueError("No completed runs found")
    result = (
        pl.concat(frames)
        .group_by("map", "time_step")
        .agg(
            pl.col(args.metric).median().alias("median"),
            pl.col(args.metric).quantile(0.25, interpolation="linear").alias("q25"),
            pl.col(args.metric).quantile(0.75, interpolation="linear").alias("q75"),
            pl.col("seed").n_unique().alias("n_seeds"),
        )
        .sort("map", "time_step")
        .collect()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.output)


if __name__ == "__main__":
    main()
