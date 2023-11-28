import torch
import numpy as np
import scipy.stats as sp
import polars as pl
from typing import Optional


class RunningMeanStd:
    """Credits to https://github.com/jcwleo/random-network-distillation-pytorch"""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, batch: torch.Tensor):
        batch_count = len(batch)
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0)
        while len(batch_mean.shape) > len(self.mean.shape):
            batch_mean = torch.mean(batch_mean, dim=0)
            batch_var = torch.var(batch_var, dim=0)

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, batch: torch.Tensor) -> torch.Tensor:
        return (batch - self.mean) / torch.sqrt(self.var + 1e-8)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / (self.count + batch_count)

        self.var = M2 / (self.count + batch_count)
        self.mean = self.mean + delta * batch_count / tot_count
        self.count = batch_count + self.count

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self


def round_col(df: pl.DataFrame, col_name: str, round_value: int):
    col = df[col_name] / round_value
    col = col.round(0)
    col = col * round_value
    col = col.cast(pl.Int64)
    return df.with_columns(col.alias(col_name))


def stats_by(col_name: str, df: pl.DataFrame):
    grouped = df.groupby(col_name)
    cols = [col for col in df.columns if col not in ["time_step", "timestamp_sec"]]
    res = grouped.agg(
        [pl.mean(col).alias(f"mean_{col}") for col in cols]
        + [pl.std(col).alias(f"std_{col}") for col in cols]
        + [pl.min(col).alias(f"min_{col}") for col in cols]
        + [pl.max(col).alias(f"max_{col}") for col in cols]
    ).sort("time_step")

    # Compute confidence intervals for 95% confidence
    # https://stackoverflow.com/questions/28242593/correct-way-to-obtain-confidence-interval-with-scipy
    confidence_intervals = []
    counts = grouped.count().sort("time_step")["count"]
    scale = (counts**0.5).to_numpy().astype(np.float32)
    for col in cols:
        mean = res[f"mean_{col}"].to_numpy().astype(np.float32)
        # Avoid zero std, otherwise a "inf" * 0 will be computed by scipy, leading to NaN
        std = res[f"std_{col}"].to_numpy().astype(np.float32) + 1e-8
        # Use scipy.stats.t if the sample size is small (then degree of freedom, df, is n_samples - 1)
        # Use scipy.stats.norm if the sample size is large
        lower, upper = sp.norm.interval(0.95, loc=mean, scale=std / scale)
        ci95 = (upper - lower) / 2
        new_col = pl.Series(name=f"ci95_{col}", values=ci95)
        # new_col = 0.95 * res[f"std_{col}"] / n_samples**0.5
        # new_col = new_col.alias(f"ci95_{col}")
        confidence_intervals.append(new_col)

    res = res.with_columns(confidence_intervals)
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
