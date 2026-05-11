import shutil
from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import typed_argparse as tap

import marl
from marl.models import Dataset

plt.rcParams.update(
    {
        "text.usetex": shutil.which("latex") is not None,  # Use latex if available
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
    }
)


class SingleLogdir(tap.TypedArgs):
    pass


class CompareLogdirs(tap.TypedArgs):
    logdirs: list[Path]
    metrics: Collection[str] | str


def plot_manually(logdir: Path):
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
    exp = marl.Experiment.load(logdir)
    datasets = exp.get_results_datasets(1000, metrics=["exit_rate", "loss"])
    for dataset in datasets:
        save_to = None
        if save:
            save_to = Path("plots") / f"{dataset.label}.pdf"
        plot(dataset, show=True, save_to=save_to)
        plt.clf()


def plot(dataset: Dataset, prefix: str = "", show=False, save_to: str | Path | None = None):
    label = f"{prefix}{dataset.nice_label} ({dataset.category})"
    plt.plot(dataset.ticks, dataset.mean, label=label)
    plt.fill_between(dataset.ticks, dataset.mean - dataset.ci95, dataset.mean + dataset.ci95, alpha=0.2)
    if show or save_to is not None:
        plt.xlabel("Time step")
        plt.ylabel(dataset.nice_label)
        plt.legend()
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(exist_ok=True)
        plt.savefig(destination)
    if show:
        plt.legend()
        plt.show()


def compare_multiple_experiments(logdirs: Collection[Path], metrics: Collection[str] | str):
    experiments = [marl.Experiment.load(logdir) for logdir in logdirs]
    datasets_dict = {exp.logdir: exp.get_results_datasets(1000, metrics=metrics) for exp in experiments}
    all_labels = set([ds.label for datasets in datasets_dict.values() for ds in datasets])
    for label in all_labels:
        for logdir, datasets in datasets_dict.items():
            [plot(ds, f"{logdir[5:]}-") for ds in datasets if ds.label == label]
        plt.legend()
        destination = Path("plots") / f"{label}.pdf"
        destination.parent.mkdir(exist_ok=True)
        plt.savefig(destination)
        plt.show()
        plt.clf()


def main():
    # plot_manually()
    logdir1 = Path("logs/vdn-False-LLE-lvl6")
    logdir2 = Path("logs/LLE-lvl6-VDN-old")
    plot_with_datasets(logdir1)
    compare_multiple_experiments([logdir1, logdir2], "loss")


if __name__ == "__main__":
    main()
