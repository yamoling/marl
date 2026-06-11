from pathlib import Path
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np

from marl import Dataset, LightExperiment

from ._setup import GlobalConfig, _setup


def plot(dataset: Dataset, prefix: str = "", show=False, save_to: str | Path | None = None):
    label = f"{prefix}{dataset.nice_label} ({dataset.category})"
    plt.plot(dataset.ticks, dataset.mean, label=label)
    low_bound = np.maximum(dataset.mean - dataset.ci95, dataset.min)
    high_bound = np.minimum(dataset.mean + dataset.ci95, dataset.max)
    plt.fill_between(dataset.ticks, low_bound, high_bound, alpha=0.2)
    if show or save_to is not None:
        plt.xlabel("Time step")
        plt.ylabel(dataset.nice_label)
        # plt.legend()
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(exist_ok=True)
        plt.savefig(destination)
    if show:
        plt.margins(x=0.01, y=0.01)
        plt.show()


def plot_learning_curves(logdirs: Collection[str], metrics: Collection[str], *, global_config: GlobalConfig = {}):
    _setup(**global_config)
    experiments = [LightExperiment.load(logdir) for logdir in logdirs]
    datasets_dict = {exp.logdir: exp.get_results_datasets(1000, metrics=metrics) for exp in experiments}
    all_labels = set([ds.label for datasets in datasets_dict.values() for ds in datasets])
    for label in all_labels:
        for logdir, datasets in datasets_dict.items():
            [plot(ds, f"{logdir[5:]}-") for ds in datasets if ds.label == label]
        # Very small legend above the plot
        plt.legend(loc="upper left", fontsize="small", bbox_to_anchor=(0, 1.02))
        plt.xlabel("Time step")
        plt.ylabel(label)
        plt.margins(x=0.01, y=0.01)
        destination = Path("plots") / f"{label}.pdf"
        destination.parent.mkdir(exist_ok=True)
        plt.savefig(destination)
        plt.show()
        plt.clf()
