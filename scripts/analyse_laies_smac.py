"""Aggregate and plot the results of `train_laies_smac.py`.

Produces the counterpart of Figure 4 of the LAIES paper: the test win rate against the number of
training steps, averaged over the seeds, with a +/- one standard deviation band.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import typed_argparse as tap

LOGGER = logging.getLogger(__name__)

# Slots 1-3 of the validated categorical palette (light mode).
CONDITION_STYLE = {
    "laies": ("LAIES (sparse)", "#2a78d6"),
    "laies-noanneal": ("LAIES, no anneal", "#eb6834"),
    "qmix-dr": ("QMIX-DR (dense)", "#1baf7a"),
    "qmix": ("QMIX (sparse)", "#eda100"),
}


class Args(tap.TypedArgs):
    logdir: str = tap.arg("--logdir", default="logs/laies-paper", help="Root directory of the experiment.")
    extra_logdir: str = tap.arg("--extra-logdir", default="", help="Second root scanned for missing conditions.")
    map_name: str = tap.arg("--map", default="2m_vs_1z")
    output: str = tap.arg("--output", default=".agents/reports/laies-2m_vs_1z.png")


def read_condition(condition_dir: Path) -> pl.DataFrame | None:
    """
    Return the per-seed test win rate of one condition, indexed by time step.

    Each `test.csv` holds one row per test episode with a boolean `battle_won` column, so the win rate
    of an evaluation point is the mean of that column over the episodes sharing the same `time_step`.

    @ai-generated
    """
    frames = []
    for run_dir in sorted(condition_dir.glob("run-*")):
        test_file = run_dir / "test.csv"
        if not test_file.exists():
            continue
        frame = (
            pl.read_csv(test_file)
            .group_by("time_step")
            .agg(pl.col("battle_won").mean().alias("win_rate") * 100)
            .with_columns(pl.lit(run_dir.name).alias("run"))
        )
        frames.append(frame)
    if not frames:
        return None
    return pl.concat(frames).sort("time_step")


def summarise(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Average the win rate over the seeds at every evaluation point.

    @ai-generated
    """
    return (
        frame.group_by("time_step")
        .agg(
            pl.col("win_rate").mean().alias("mean"),
            pl.col("win_rate").std().fill_null(0.0).alias("std"),
            pl.len().alias("n_seeds"),
        )
        .sort("time_step")
    )


def style_axes(axes, title: str):
    """
    Apply the shared chart chrome: recessive grid, no top/right spines, muted axis colours.

    @ai-generated
    """
    axes.set_xlabel("T (mil)")
    axes.set_ylabel("Test Win %")
    axes.set_title(title, fontsize=10, color="#0b0b0b")
    axes.set_ylim(-2, 102)
    axes.grid(True, color="#e5e5e2", linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors="#52514e", labelsize=9)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color("#c3c2b7")


def plot_per_seed(root: Path, output: Path, args: Args):
    """
    Draw the per-seed curves of the two LAIES variants side by side.

    Identity here is carried by the facet, not by colour, so each panel uses a single hue and every
    seed is a thin line. This is what makes the annealed variant's rise-then-collapse visible: the
    seed-averaged curve of the main figure hides that every seed peaks at a different time.

    @ai-generated
    """
    facets = [
        ("laies", "LAIES — intrinsic rewards annealed", "#2a78d6"),
        ("laies-noanneal", "LAIES — no annealing", "#eb6834"),
    ]
    figure, all_axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for axes, (condition, title, colour) in zip(all_axes, facets):
        raw = read_condition(root / condition)
        if raw is None:
            continue
        for run in sorted(raw["run"].unique()):
            seed_frame = raw.filter(pl.col("run") == run).sort("time_step")
            axes.plot(
                seed_frame["time_step"].to_numpy() / 1e6,
                seed_frame["win_rate"].to_numpy(),
                color=colour,
                linewidth=1.2,
                alpha=0.75,
            )
        summary = summarise(raw)
        axes.plot(
            summary["time_step"].to_numpy() / 1e6,
            summary["mean"].to_numpy(),
            color=colour,
            linewidth=2.5,
        )
        style_axes(axes, title)
    all_axes[1].set_ylabel("")
    figure.suptitle("Per-seed test win rate (thin: one seed, thick: mean of 6)", fontsize=10, color="#52514e")
    figure.tight_layout()
    figure.savefig(output, dpi=160, facecolor="#fcfcfb")
    print(f"Per-seed figure written to {output}")


def main(args: Args):
    """
    Plot every available condition and print a summary table of the final win rates.

    @ai-generated
    """
    root = Path(args.logdir) / args.map_name
    figure, axes = plt.subplots(figsize=(7.6, 4.4))
    summaries = {}
    endpoints = []
    for condition, (label, colour) in CONDITION_STYLE.items():
        raw = read_condition(root / condition)
        if raw is None and args.extra_logdir:
            raw = read_condition(Path(args.extra_logdir) / args.map_name / condition)
        if raw is None:
            LOGGER.warning("No data for condition %s", condition)
            continue
        summary = summarise(raw)
        summaries[condition] = summary
        steps = summary["time_step"].to_numpy() / 1e6
        mean = summary["mean"].to_numpy()
        std = summary["std"].to_numpy()
        axes.plot(steps, mean, color=colour, linewidth=2, label=label)
        axes.fill_between(steps, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0)
        endpoints.append((float(mean[-1]), float(steps[-1]), label))

    # Direct labels: two slots of the palette are below 3:1 contrast on a light surface, so identity
    # never relies on colour alone. Curves that end at the same value (here the two 0% baselines)
    # would print on top of each other, so the labels are pushed apart by a minimum vertical gap.
    min_gap = 6.0
    placed = None
    for value, step, label in sorted(endpoints):
        placed = value if placed is None else max(value, placed + min_gap)
        axes.annotate(
            label,
            (step, value),
            xytext=(step + 0.012, placed),
            textcoords="data",
            color="#52514e",
            fontsize=9,
            va="center",
            annotation_clip=False,
        )

    style_axes(axes, f"SMAC {args.map_name} — test win rate, mean of 6 seeds ± 1 std")
    axes.legend(frameon=False, loc="upper left", fontsize=9)
    figure.subplots_adjust(right=0.74)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="#fcfcfb")
    print(f"Figure written to {output}")
    plot_per_seed(root, output.with_name(f"{output.stem}-seeds{output.suffix}"), args)

    print(f"\n{'condition':<12}{'seeds':>6}{'final win %':>14}{'best win %':>13}{'steps':>10}")
    for condition, summary in summaries.items():
        last = summary.tail(1)
        best = summary["mean"].max()
        print(
            f"{condition:<12}{last['n_seeds'][0]:>6}"
            f"{last['mean'][0]:>10.1f} ± {last['std'][0]:<4.1f}"
            f"{best:>13.1f}{last['time_step'][0]:>10}"
        )

    # Table view (the relief rule for the low-contrast slot) as a CSV next to the figure.
    table = None
    for condition, summary in summaries.items():
        renamed = summary.select(
            "time_step",
            pl.col("mean").round(2).alias(f"{condition}_mean"),
            pl.col("std").round(2).alias(f"{condition}_std"),
        )
        table = renamed if table is None else table.join(renamed, on="time_step", how="full", coalesce=True)
    if table is not None:
        table_path = output.with_suffix(".csv")
        table.sort("time_step").write_csv(table_path)
        print(f"Table written to {table_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    tap.Parser(Args).bind(main).run()
