"""
Aggregate and plot the results of `train_acer_gym.py` (and of any other ACER study whose
configurations are stored as `logs/<study>/<config>/run-<seed>/test.csv`).

The layout follows Figure 1 of the ACER paper: one colour per replay ratio, and the trust region
variant is encoded with the line style (solid = trust region, dashed = no trust region) rather than
with a ninth colour.
"""

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import typed_argparse as tap

LOGGER = logging.getLogger(__name__)

# Slots 1, 2, 3 and 7 of the validated categorical palette (light mode), which pass the all-pairs
# colour-vision-deficiency checks together. Used for the four replay ratios.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
# Slots 1 to 6 of the same palette, in their documented order, for studies whose configurations are
# not replay ratios. Validated on the adjacent pairlist, which is the one that applies to line charts.
SERIES_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
LABELS = {
    "iacer": "IACER",
    "iacer-onpolicy": "IACER (no replay)",
    "macer-vdn": "MACER-VDN",
    "macer-qmix": "MACER-QMix",
    "ippo": "IPPO",
    "mappo-qmix": "MAPPO-QMix",
    "acer": "ACER",
    "no-trust-region": "no trust region",
    "no-truncation": "no truncation",
    "one-step-target": "one-step target",
    "uncorrected-q-lambda": "uncorrected Q(lambda)",
}
ORDER = tuple(LABELS)
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


class Args(tap.TypedArgs):
    logdir: Path = tap.arg("--logdir", default=Path("logs/acer-replay/CartPole-v1"))
    metric: str = tap.arg("--metric", default="score-0", help="Column of test.csv to aggregate.")
    output: Path = tap.arg("--output", default=Path(".agents/reports/acer-replay-cartpole.png"))
    max_step: int = tap.arg("--max-step", default=0, help="Only keep the first N steps (0 = all).")
    per_seed: bool = tap.arg("--per-seed", default=False, help="Small multiples with one line per seed.")
    title: str = tap.arg("--title", default="")


def read_config(config_dir: Path, metric: str) -> pl.DataFrame | None:
    """
    Per-seed learning curve of one configuration: the mean of `metric` over the test episodes of every
    evaluation point, for every run of the configuration.

    @ai-generated
    """
    frames = []
    for run_dir in sorted(config_dir.glob("run-*"), key=lambda d: int(d.name.split("-")[-1])):
        test_file = run_dir / "test.csv"
        if not test_file.exists():
            continue
        try:
            frame = pl.read_csv(test_file)
        except pl.exceptions.NoDataError:
            # The run has not reached its first evaluation point yet.
            continue
        if metric not in frame.columns:
            LOGGER.warning("%s has no column %s", test_file, metric)
            continue
        frames.append(
            frame.group_by("time_step")
            .agg(pl.col(metric).mean().alias("value"))
            .with_columns(pl.lit(run_dir.name).alias("run"))
        )
    if len(frames) == 0:
        return None
    return pl.concat(frames)


def facet(args: "Args", per_seed: dict[str, pl.DataFrame], curves: dict[str, pl.DataFrame]):
    """
    Small multiples: one panel per configuration showing every seed as a thin line and their mean as a
    thick one.

    The seeds of a run are often bimodal (a task is either solved or not solved at all), and a
    mean +/- standard deviation band then describes a performance that no seed ever reached. Plotting
    the seeds individually shows how many of them actually succeeded.

    @ai-generated
    """
    n_cols = min(4, len(curves))
    n_rows = math.ceil(len(curves) / n_cols)
    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    for index, (name, frame) in enumerate(curves.items()):
        colour, _, label = style_of(name, index)
        cell = axes[index // n_cols][index % n_cols]
        for run, group in per_seed[name].group_by("run"):
            group = group.sort("time_step")
            cell.plot(group["time_step"], group["value"], color=colour, linewidth=0.9, alpha=0.45)
        cell.plot(frame["time_step"], frame["mean"], color=colour, linewidth=2)
        cell.set_title(label, color=TEXT_PRIMARY, fontsize=10)
        cell.grid(True, color="#e6e5e2", linewidth=0.8)
        cell.set_axisbelow(True)
        for spine in ("top", "right"):
            cell.spines[spine].set_visible(False)
        cell.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for index in range(len(curves), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].set_visible(False)
    figure.supxlabel("Environment steps", color=TEXT_SECONDARY, fontsize=10)
    figure.supylabel(args.metric, color=TEXT_SECONDARY, fontsize=10)
    figure.suptitle(args.title or args.logdir.as_posix(), color=TEXT_PRIMARY)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


def aggregate(per_seed: pl.DataFrame) -> pl.DataFrame:
    """
    Mean and standard deviation of the learning curve across the seeds.

    @ai-generated
    """
    return (
        per_seed.group_by("time_step")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std().fill_null(0.0).alias("std"),
            pl.col("run").n_unique().alias("n_seeds"),
        )
        .sort("time_step")
    )


def style_of(config_name: str, index: int) -> tuple[str, str, str]:
    """
    Colour, line style and legend label of a configuration.

    Configurations of the replay study are named `replay-<ratio>-<trust|no-trust>`: the ratio picks the
    colour and the trust region picks the line style. Any other configuration falls back to the
    categorical order of the palette.

    @ai-generated
    """
    parts = config_name.split("-")
    if parts[0] == "replay" and len(parts) >= 3:
        ratios = ("0", "1", "4", "8")
        ratio = parts[1]
        trust = "-".join(parts[2:]) == "trust"
        colour = PALETTE[ratios.index(ratio)]
        label = f"ratio {ratio}" + ("" if trust else ", no TR")
        return colour, "-" if trust else "--", label
    if index >= len(SERIES_PALETTE):
        raise ValueError(
            f"{index + 1} configurations exceed the {len(SERIES_PALETTE)} validated colours: "
            "split the study into several figures instead of cycling the palette."
        )
    return SERIES_PALETTE[index], "-", LABELS.get(config_name, config_name)


def spread_labels(labels: list[tuple[float, float, str]], axes) -> list[tuple[float, float, str]]:
    """
    Push apart the direct labels placed at the end of the lines so that converged curves do not print
    their names on top of each other.

    @ai-generated
    """
    labels = sorted(labels)
    span = max(y for y, _, _ in labels) - min(y for y, _, _ in labels)
    min_gap = 0.04 * (span if span > 0 else 1.0)
    spread = []
    previous = None
    for y, x_end, label in labels:
        if previous is not None and y - previous < min_gap:
            y = previous + min_gap
        spread.append((y, x_end, label))
        previous = y
    return spread


def plot(args: Args, curves: dict[str, pl.DataFrame]):
    """
    Draw one line per configuration with a +/- one standard deviation band and a direct label at the
    end of each line, then save the figure to `args.output`.

    @ai-generated
    """
    figure, axes = plt.subplots(figsize=(9, 5.5))
    end_labels = []
    for index, (name, frame) in enumerate(curves.items()):
        colour, line_style, label = style_of(name, index)
        x = frame["time_step"].to_numpy()
        mean = frame["mean"].to_numpy()
        std = frame["std"].to_numpy()
        axes.plot(x, mean, color=colour, linestyle=line_style, linewidth=2, label=label)
        axes.fill_between(x, mean - std, mean + std, color=colour, alpha=0.12, linewidth=0)
        end_labels.append((float(mean[-1]), float(x[-1]), label))
    for y, x_end, label in spread_labels(end_labels, axes):
        axes.annotate(
            label,
            (x_end, y),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=8,
            va="center",
        )
    axes.set_xlabel("Environment steps", color=TEXT_SECONDARY)
    axes.set_ylabel(args.metric, color=TEXT_SECONDARY)
    axes.set_title(args.title or args.logdir.as_posix(), color=TEXT_PRIMARY)
    axes.grid(True, color="#e6e5e2", linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.tick_params(colors=TEXT_SECONDARY)
    axes.legend(frameon=False, loc="upper left", fontsize=9, ncols=2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


def truncate_to_common_budget(curves: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """
    Restrict every learning curve to the number of steps that all the configurations have reached.

    Configurations with a high replay ratio are slower in wall-clock time, so comparing them while the
    study is still running would otherwise compare curves of different lengths.

    @ai-generated
    """
    budgets = [frame["time_step"].max() for frame in curves.values()]
    budget = min(b for b in budgets if b is not None)
    if budget < max(b for b in budgets if b is not None):
        print(f"Truncating every curve to the common budget of {budget} steps.")
    return {name: frame.filter(pl.col("time_step") <= budget) for name, frame in curves.items()}


def summarize(curves: dict[str, pl.DataFrame], metric: str):
    """
    Print the table view of the figure: the number of seeds, the score averaged over the last 20% of
    the training steps (final performance) and the score averaged over the whole run (a proxy for the
    sample efficiency, i.e. the area under the learning curve).

    @ai-generated
    """
    print(f"{'configuration':<28}{'seeds':>6}{'final ' + metric:>18}{'AUC ' + metric:>18}")
    for name, frame in curves.items():
        last_steps = frame["time_step"].max()
        assert last_steps is not None
        final = frame.filter(pl.col("time_step") >= 0.8 * last_steps)["mean"].mean()
        auc = frame["mean"].mean()
        n_seeds = frame["n_seeds"].max()
        print(f"{name:<28}{n_seeds:>6}{final:>18.1f}{auc:>18.1f}")


def main(args: Args):
    """
    Read every configuration of the study, print the summary table and save the figure.

    @ai-generated
    """
    curves = dict[str, pl.DataFrame]()
    seed_curves = dict[str, pl.DataFrame]()
    directories = sorted(
        (d for d in args.logdir.iterdir() if d.is_dir()),
        key=lambda d: (ORDER.index(d.name) if d.name in ORDER else len(ORDER), d.name),
    )
    for config_dir in directories:
        if not config_dir.is_dir():
            continue
        per_seed = read_config(config_dir, args.metric)
        if per_seed is None:
            LOGGER.warning("No results in %s", config_dir)
            continue
        seed_curves[config_dir.name] = per_seed
        curves[config_dir.name] = aggregate(per_seed)
    if len(curves) == 0:
        raise SystemExit(f"No results found in {args.logdir}")
    if args.max_step > 0:
        curves = {name: frame.filter(pl.col("time_step") <= args.max_step) for name, frame in curves.items()}
    curves = truncate_to_common_budget(curves)
    summarize(curves, args.metric)
    if args.per_seed:
        seed_curves = {
            name: seed_curves[name].filter(pl.col("time_step") <= frame["time_step"].max())
            for name, frame in curves.items()
        }
        facet(args, seed_curves, curves)
    else:
        plot(args, curves)


if __name__ == "__main__":
    tap.Parser(Args).bind(main).run()
