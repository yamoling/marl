import shutil
from pathlib import Path

import matplotlib.pyplot as plt

import marl

plt.rcParams.update(
    {
        "text.usetex": shutil.which("latex") is not None,  # Use latex if available
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
    }
)
LOGDIR = Path("logs/VDN-LLE-lvl6")


def plot_manually():
    exp = marl.Experiment.load(LOGDIR)
    df = exp.get_test_results(granularity=1000).collect()
    print(df)
    columns = [col[5:] for col in df.columns if col != "ticks" and col.startswith("mean-")]
    x = df["ticks"]
    destination = Path(exp.logdir, "plots")
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


def plot_with_datasets():
    exp = marl.Experiment.load(LOGDIR)
    datasets = exp.get_results_datasets(1000, metrics=["exit_rate", "loss"])
    for dataset in datasets:
        plt.plot(dataset.ticks, dataset.mean, label=f"{dataset.label} ({dataset.category})")
        plt.fill_between(dataset.ticks, dataset.mean - dataset.ci95, dataset.mean + dataset.ci95, alpha=0.2)
        plt.xlabel("Time step")
        plt.ylabel(dataset.label.capitalize())
        plt.legend()
        plt.show()


def main():
    # plot_manually()
    plot_with_datasets()


if __name__ == "__main__":
    main()
