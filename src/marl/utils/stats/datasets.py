import re
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import scipy.stats as sp
import polars as pl
import polars.exceptions as pl_errors
from typing import Optional


@dataclass
class Dataset:
    logdir: str
    ticks: list[int]
    label: str
    category: str
    mean: npt.NDArray[np.float32]
    min: npt.NDArray[np.float32]
    max: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]
    ci95: npt.NDArray[np.float32]


@dataclass
class ExperimentResults:
    logdir: str
    datasets: list[Dataset]
    qvalue_ds: list[Dataset]


def _concat_with_missing_columns(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate frames with potentially different schemas by padding missing columns with nulls."""
    if len(dfs) == 0:
        return pl.DataFrame()
    all_columns = sorted(set().union(*(df.columns for df in dfs)))
    aligned = []
    for df in dfs:
        missing = [col for col in all_columns if col not in df.columns]
        if len(missing) > 0:
            df = df.with_columns([pl.lit(None).alias(col) for col in missing])
        aligned.append(df.select(all_columns))
    return pl.concat(aligned)


def round_col(df: pl.DataFrame, col_name: str, round_value: int):
    """
    Round the values of `col_name` to the closest multiple of `round_value`.

    This is particularly useful to map each training time stet to the closest test interval, in order to compute the average test metrics for each training time step.
    """
    if round_value == 0:
        raise ValueError("round_value must be different from 0")
    try:
        col = df[col_name] / round_value
        col = col.round(0)
        col = col * round_value
        col = col.cast(pl.Int64)
        return df.with_columns(col.alias(col_name))
    except pl_errors.ColumnNotFoundError:
        return df


def stats_by(col_name: str, df: pl.DataFrame, replace_inf: bool):
    if len(df) == 0:
        return df
    grouped = df.group_by(col_name)
    cols = [col for col in df.columns if col != col_name]
    res = grouped.agg(
        [pl.mean(col).alias(f"mean-{col}") for col in cols]
        + [pl.std(col).alias(f"std-{col}") for col in cols]
        + [pl.min(col).alias(f"min-{col}") for col in cols]
        + [pl.max(col).alias(f"max-{col}") for col in cols]
    ).sort(col_name)

    # Compute confidence intervals for 95% confidence
    # https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
    confidence_intervals = []
    counts = grouped.len().sort(col_name)["len"]
    scale = (counts**0.5).to_numpy().astype(np.float32)
    for col in cols:
        mean = res[f"mean-{col}"].to_numpy().astype(np.float32)
        # Avoid zero std with +1e-8, otherwise a "inf" * 0 will be computed by scipy, leading to NaN
        std = res[f"std-{col}"].to_numpy().astype(np.float32) + 1e-8
        # Use scipy.stats.t if the sample size is small (then degree of freedom, df, is n_samples - 1)
        # Use scipy.stats.norm if the sample size is large
        lower, upper = sp.norm.interval(0.95, loc=mean, scale=std / scale)
        ci95 = (upper - lower) / 2
        new_col = pl.Series(name=f"ci95-{col}", values=ci95)
        # new_col = 0.95 * res[f"std_{col}"] / n_samples**0.5
        # new_col = new_col.alias(f"ci95_{col}")
        confidence_intervals.append(new_col)

    res = res.with_columns(confidence_intervals)
    if replace_inf:
        for series in res.select(pl.selectors.float()):
            mask = series.is_infinite() | series.is_nan()
            series[mask] = 1e20
            res = res.with_columns(series)
            # Formerly:
            # res.replace(series.name, series)
    return res


def compute_dataframe(dfs: list[pl.DataFrame], replace_inf: bool) -> pl.DataFrame:
    dfs = [d for d in dfs if not d.is_empty()]
    if len(dfs) == 0:
        return pl.DataFrame()
    df = _concat_with_missing_columns(dfs)
    if "timestamp_sec" in df.columns:
        df = df.drop("timestamp_sec")
    to_drop = list[str]()
    for col, dtype in zip(df.columns, df.dtypes):
        if not dtype.is_numeric():
            to_drop.append(col)
    df = df.drop(to_drop)
    score_cols = [col for col in df.columns if col.startswith("score")]
    if len(score_cols) != 0:
        df = df.with_columns(score=pl.sum_horizontal(score_cols))
    return stats_by("time_step", df, replace_inf)


def compute_datasets(
    dfs: list[pl.DataFrame],
    logdir: str,
    replace_inf: bool,
    category: str,
) -> list[Dataset]:
    """
    Aggregates dataframes and computes the stats (mean, std, etc) for each column.

    Returns the list of datasets, one for each metric column in the dataframes.
    """
    dfs = [d for d in dfs if not d.is_empty()]
    if len(dfs) == 0:
        return []
    df = _concat_with_missing_columns(dfs)
    if "timestamp_sec" in df.columns:
        df = df.drop("timestamp_sec")
    score_cols = [col for col in df.columns if col.startswith("score")]
    if len(score_cols) != 0:
        df = df.with_columns([pl.sum_horizontal(score_cols).alias("score-sum")])
    to_drop = list[str]()
    for col, dtype in zip(df.columns, df.dtypes):
        if not dtype.is_numeric():
            to_drop.append(col)
    df = df.drop(to_drop)
    df_stats = stats_by("time_step", df, replace_inf)
    res = list[Dataset]()
    ticks = df_stats["time_step"].to_list()
    for col in df.columns:
        if col == "time_step":
            continue
        res.append(
            Dataset(
                logdir=logdir,
                ticks=ticks,
                label=col,
                category=category,
                mean=df_stats[f"mean-{col}"].to_numpy().astype(np.float32),
                std=df_stats[f"std-{col}"].to_numpy().astype(np.float32),
                min=df_stats[f"min-{col}"].to_numpy().astype(np.float32),
                max=df_stats[f"max-{col}"].to_numpy().astype(np.float32),
                ci95=df_stats[f"ci95-{col}"].to_numpy().astype(np.float32),
            )
        )
    return res


def compute_qvalues(dfs: list[pl.DataFrame], logdir: str, replace_inf: bool, reward_components: list[str], n_agents: int) -> list[Dataset]:
    """
    Aggregates qvalues"""
    dfs = [d for d in dfs if not d.is_empty()]
    if len(dfs) == 0:
        return []
    df = _concat_with_missing_columns(dfs)
    if "timestamp_sec" in df.columns:
        df = df.drop("timestamp_sec")

    df_stats = stats_by("time_step", df, replace_inf)
    res = list[Dataset]()
    ticks = df_stats["time_step"].to_list()

    l_metrics = ["mean-", "std-", "min-", "max-", "ci95-"]
    for i in range(n_agents):
        prefix = f"agent{i}"
        for metric in l_metrics:
            selected_columns = df_stats.select(pl.selectors.contains(f"{metric}{prefix}").abs())
            row_sum = pl.sum_horizontal(selected_columns)
            df_stats = df_stats.with_columns(
                **{col: (pl.col(col) / row_sum) for col in selected_columns.columns}
            )  # Normalize over qvalue type
    for col in df.columns:
        if col == "time_step":
            continue
        col_title = col.split("-")
        if "qvalue" in col_title[1] and len(reward_components) > 1:
            label = f"{col_title[0]}-{reward_components[int(re.sub(r'\D', '', col_title[1]))]}"
        else:
            label = f"{col_title[0]}-{reward_components[0]}"
        res.append(
            Dataset(
                logdir,
                category="Q-values",
                ticks=ticks,
                label=label,
                mean=df_stats[f"mean-{col}"].to_numpy().astype(np.float32),
                std=df_stats[f"std-{col}"].to_numpy().astype(np.float32),
                min=df_stats[f"min-{col}"].to_numpy().astype(np.float32),
                max=df_stats[f"max-{col}"].to_numpy().astype(np.float32),
                ci95=df_stats[f"ci95-{col}"].to_numpy().astype(np.float32),
            )
        )
    return res


def agregate_metrics(
    all_metrics: list[dict[str, float]],
    only_avg=False,
    skip_keys: Optional[set[str]] = None,
) -> dict[str, float]:
    """Aggregate a list of metrics into min, max, avg and std."""
    import numpy as np

    if skip_keys is None:
        skip_keys = set()
    all_values: dict[str, list[float]] = {}
    for metrics in all_metrics:
        for key, value in metrics.items():
            if key not in all_values:
                all_values[key] = []
            all_values[key].append(value)
    res = {}
    if only_avg:
        for key, values in all_values.items():
            res[key] = float(np.average(np.array(values)))
    else:
        for key, values in all_values.items():
            if key not in skip_keys:
                values = np.array(values)
                res[f"avg_{key}"] = float(np.average(values))
                res[f"std_{key}"] = float(np.std(values))
                res[f"min_{key}"] = float(values.min())
                res[f"max_{key}"] = float(values.max())
    return res


def moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    data_shape = x.shape
    ones = np.ones_like((window_size, *data_shape[1:]))
    return np.convolve(x, ones, "valid") / window_size


def ensure_numerical(df: pl.DataFrame, drop_non_numeric: bool = True):
    non_numerical = [col for col in df.select(~pl.selectors.numeric()).columns]
    to_drop = []
    for col in non_numerical:
        try:
            df = df.cast({col: pl.Float32})
        except pl.exceptions.InvalidOperationError:
            to_drop.append(col)
    if drop_non_numeric:
        df = df.drop(to_drop)
    return df
