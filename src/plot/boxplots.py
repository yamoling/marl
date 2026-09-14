import logging
from collections.abc import Collection
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl

from marl import LightExperiment
from marl.logging import TIME_STEP_COL

from ._setup import GlobalConfig, _setup
from .utils import deslugify

logger = logging.getLogger(__name__)


def boxplot_at(
    experiments: Collection[LightExperiment],
    metric: str,
    /,
    t: int | Literal["mean", "first", "last"] = "last",
    kind: Literal["Train", "Test", "Training data"] = "Test",
    *,
    global_config: GlobalConfig = None,
):
    if global_config is None:
        global_config = {}
    _setup(**global_config)
    results = []
    labels = []
    for e in experiments:
        match kind:
            case "Train":
                dfs = [r.train_metrics for r in e.runs]
            case "Test":
                dfs = [r.test_metrics for r in e.runs]
            case "Training data":
                dfs = [r.training_data for r in e.runs]
            case _:
                raise ValueError(f"Invalid kind: {kind}")
        dfs = [df.select(pl.col(TIME_STEP_COL), pl.col(metric)) for df in dfs]
        match t:
            case "mean":
                dfs = [df.mean() for df in dfs]
            case "first":
                dfs = [df.first() for df in dfs]
            case "last":
                dfs = [df.last() for df in dfs]
            case int():
                dfs = [df.filter(pl.col(TIME_STEP_COL) == t) for df in dfs]
            case _:
                raise ValueError(f"Invalid t: {t}")
        try:
            df = pl.concat(dfs).collect()
            results.append(df[metric].to_numpy())
            # Remove "logs/" prefix
            labels.append(e.logdir[5:])
        except pl.exceptions.PolarsError:
            logger.warning(f"An error occurred while processing {e.logdir} with {metric}. Skipping.")

    fig, ax = plt.subplots()
    ax.boxplot(results)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(deslugify(metric))
    ax.set_xlabel("Experiments")
    return fig, ax
