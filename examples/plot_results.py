import argparse
import shutil
from collections.abc import Collection
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import marl
from marl.models import Dataset

plt.rcParams.update(
    {
        "text.usetex": shutil.which("latex") is not None,  # Use latex if available
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
    }
)


def plot_manually(logdir: Path):
    """Plot aggregated test metrics from a single experiment."""
    exp = marl.Experiment.load(logdir)
    df = exp.get_test_results(granularity=1000).collect()
    columns = [col[5:] for col in df.columns if col != "ticks" and col.startswith("mean-")]
    x = df["ticks"]
    destination = logdir / "plots"
    destination.mkdir(exist_ok=True)
    for col in columns:
        y = df[f"mean-{col}"]
        ci95 = df[f"ci95-{col}"]
        plt.plot(x, y, label=col)
        plt.fill_between(x, y - ci95, y + ci95, alpha=0.2)
        plt.xlabel("Time step")
        plt.ylabel(col)
        plt.savefig(destination / f"{col}.pdf")
        plt.show()
        plt.clf()


def plot_with_datasets(logdir: Path, save: bool = True):
    """Plot selected experiment datasets without overwriting other categories."""
    exp = marl.Experiment.load(logdir)
    datasets = exp.get_results_datasets(1000, metrics=["exit_rate", "loss"])
    for dataset in datasets:
        save_to = Path("plots") / f"{dataset.category}-{dataset.label}.pdf" if save else None
        plot(dataset, show=True, save_to=save_to)
        plt.clf()


def plot(dataset: Dataset, prefix: str = "", show=False, save_to: str | Path | None = None):
    """Draw a dataset's mean and bounded confidence interval."""
    label = f"{prefix}{dataset.nice_label} ({dataset.category})"
    plt.plot(dataset.ticks, dataset.mean, label=label)
    low_bound = np.maximum(dataset.mean - dataset.ci95, dataset.min)
    high_bound = np.minimum(dataset.mean + dataset.ci95, dataset.max)
    plt.fill_between(dataset.ticks, low_bound, high_bound, alpha=0.2)
    if show or save_to is not None:
        plt.xlabel("Time step")
        plt.ylabel(dataset.nice_label)
        # plt.legend()
    if show or save_to is not None:
        plt.margins(x=0.01, y=0.01)
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(exist_ok=True)
        plt.savefig(destination)
    if show:
        plt.show()


def compare_multiple_experiments(logdirs: Collection[Path], metrics: Collection[str] | str):
    """Compare matching metric/category pairs across experiments."""
    experiments = [marl.Experiment.load(logdir) for logdir in logdirs]
    datasets_dict = {exp.logdir: exp.get_results_datasets(1000, metrics=metrics) for exp in experiments}
    keys = {(ds.category, ds.label) for datasets in datasets_dict.values() for ds in datasets}
    for category, label in sorted(keys):
        for logdir, datasets in datasets_dict.items():
            for dataset in datasets:
                if (dataset.category, dataset.label) == (category, label):
                    plot(dataset, f"{Path(logdir).name}-")
        plt.legend(loc="upper left", fontsize="small", bbox_to_anchor=(0, 1.02))
        plt.xlabel("Time step")
        plt.ylabel(f"{category}: {label}")
        plt.margins(x=0.01, y=0.01)
        destination = Path("plots") / f"{category}-{label}.pdf"
        destination.parent.mkdir(exist_ok=True)
        plt.savefig(destination)
        plt.show()
        plt.clf()


def main():
    """Select experiments explicitly or discover valid log directories. @ai-edited"""
    parser = argparse.ArgumentParser(description="Compare experiment result curves")
    parser.add_argument("logdirs", nargs="*", type=Path, help="experiment directories (default: logs/*)")
    parser.add_argument("--metrics", nargs="+", default=["exit_rate"], help="metric names to plot")
    args = parser.parse_args()
    logdirs = args.logdirs or [path.parent for path in Path("logs").glob("*/experiment.json")]
    if not logdirs:
        parser.error("No experiment directories found; pass logdirs explicitly")
    compare_multiple_experiments(logdirs, args.metrics)


if __name__ == "__main__":
    main()
