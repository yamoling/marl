from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import polars.exceptions as pl_errors
from typing import Sequence
from marl.logging import TickColumn, TIME_STEP_COL, TIMESTAMP_COL


@dataclass
class Dataset:
    logdir: str
    ticks: list[float]
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


def _concat_with_missing_columns(dfs: Sequence[pl.DataFrame]) -> pl.DataFrame:
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


def round_col(df: pl.LazyFrame, col_name: str, round_value: int):
    """
    Round the values of `col_name` to the closest multiple of `round_value`.

    This is particularly useful to map each training time stet to the closest test interval, in order to compute the average test metrics for each training time step.
    """
    if round_value == 0:
        raise ValueError("round_value must be different from 0")
    try:
        return df.with_columns(((pl.col(col_name) / round_value).round(0) * round_value).cast(pl.Int64))
    except pl_errors.ColumnNotFoundError:
        return df


def compute_experiment_results(dfs: Sequence[pl.LazyFrame], aggregate_by: str, granularity: int):
    dfs = [df.with_columns(run_id=pl.lit(i)) for i, df in enumerate(dfs)]
    return (
        pl.concat(dfs, how="diagonal_relaxed")
        .with_columns(
            ticks=(
                (pl.col(TIMESTAMP_COL) - pl.col(TIMESTAMP_COL).min().over(pl.len()))
                if aggregate_by == "timestamp_sec"
                else pl.col(TIME_STEP_COL)
            )
        )
        # Round ticks to granularity
        .with_columns(ticks=((pl.col("ticks") / granularity).round(0) * granularity).cast(pl.Int64))
        # First compute the mean within each run
        .group_by("ticks", "run_id")
        .mean()
        # Then compute the metrics' stats across runs
        .group_by("ticks")
        .agg(
            cs.numeric().mean().name.prefix("mean-"),
            cs.numeric().std().name.prefix("std-"),
            cs.numeric().min().name.prefix("min-"),
            cs.numeric().max().name.prefix("max-"),
            (cs.numeric().std() * 1.96 / pl.len().sqrt()).name.prefix("ci95-"),
        )
        .sort("ticks")
    )


def compute_experiment_results2(dfs: Sequence[pl.LazyFrame], aggregate_by: TickColumn, granularity: int):
    preprocessed_dfs = list[pl.LazyFrame]()
    for df in dfs:
        if aggregate_by == "timestamp_sec":
            df = df.with_columns(ticks=pl.col(TIMESTAMP_COL) - pl.col(TIMESTAMP_COL).min())
        else:
            df = df.with_columns(ticks=pl.col(TIME_STEP_COL))
        df = df.drop([TIMESTAMP_COL, TIME_STEP_COL])
        # Round the ticks to the closest multiple of granularity, and compute the granularity-wise mean
        df = df.with_columns(((pl.col("ticks") / granularity).round() * granularity).cast(pl.Int64))
        df = df.group_by("ticks").mean()
        preprocessed_dfs.append(df)
    df = pl.concat(preprocessed_dfs, how="diagonal_relaxed")
    cols = [col for col in df.collect_schema().names() if col != "ticks"]
    return (
        df.group_by("ticks")
        .agg(
            **{f"mean-{col}": pl.mean(col) for col in cols},
            **{f"std-{col}": pl.std(col) for col in cols},
            **{f"min-{col}": pl.min(col) for col in cols},
            **{f"max-{col}": pl.max(col) for col in cols},
            **{f"ci95-{col}": 1.96 * pl.std(col) / pl.count(col).sqrt() for col in cols},
        )
        .sort("ticks")
    )


def compute_qvalues(dfs: list[pl.DataFrame], logdir: str, replace_inf: bool, reward_components: list[str], n_agents: int) -> list[Dataset]:
    """Aggregates qvalues"""
    raise NotImplementedError("This function must be checked")
    # dfs = [d for d in dfs if not d.is_empty()]
    # if len(dfs) == 0:
    #     return []
    # df = _concat_with_missing_columns(dfs)
    # if "timestamp_sec" in df.columns:
    #     df = df.drop("timestamp_sec")

    # df_stats = stats_by("time_step", df, replace_inf)
    # res = list[Dataset]()
    # ticks = df_stats["time_step"].to_list()

    # l_metrics = ["mean-", "std-", "min-", "max-", "ci95-"]
    # for i in range(n_agents):
    #     prefix = f"agent{i}"
    #     for metric in l_metrics:
    #         selected_columns = df_stats.select(pl.selectors.contains(f"{metric}{prefix}").abs())
    #         row_sum = pl.sum_horizontal(selected_columns)
    #         df_stats = df_stats.with_columns(
    #             **{col: (pl.col(col) / row_sum) for col in selected_columns.columns}
    #         )  # Normalize over qvalue type
    # for col in df.columns:
    #     if col == "time_step":
    #         continue
    #     col_title = col.split("-")
    #     if "qvalue" in col_title[1] and len(reward_components) > 1:
    #         label = f"{col_title[0]}-{reward_components[int(re.sub(r'\D', '', col_title[1]))]}"
    #     else:
    #         label = f"{col_title[0]}-{reward_components[0]}"
    #     res.append(
    #         Dataset(
    #             logdir,
    #             category="Q-values",
    #             ticks=ticks,
    #             label=label,
    #             mean=df_stats[f"mean-{col}"].to_numpy().astype(np.float32),
    #             std=df_stats[f"std-{col}"].to_numpy().astype(np.float32),
    #             min=df_stats[f"min-{col}"].to_numpy().astype(np.float32),
    #             max=df_stats[f"max-{col}"].to_numpy().astype(np.float32),
    #             ci95=df_stats[f"ci95-{col}"].to_numpy().astype(np.float32),
    #         )
    #     )
    # return res


def moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    data_shape = x.shape
    ones = np.ones_like((window_size, *data_shape[1:]))
    return np.convolve(x, ones, "valid") / window_size
