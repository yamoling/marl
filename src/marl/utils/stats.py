import torch
import numpy as np
import scipy.stats as sp
import polars as pl
from dataclasses import dataclass


@dataclass
class RunningMeanStd:
    """Credits to https://github.com/openai/random-network-distillation"""

    mean: torch.Tensor
    variance: torch.Tensor
    clip_min: float
    clip_max: float

    def __init__(
        self,
        shape: tuple[int, ...] = (1,),
        clip_min: float = -5,
        clip_max: float = 5,
    ):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.variance = torch.ones(shape, dtype=torch.float32)
        self.count = 0
        self.clip_min = clip_min
        self.clip_max = clip_max

    @property
    def std(self):
        return self.variance**0.5

    def update(self, batch: torch.Tensor):
        batch_mean = torch.mean(batch, dim=0)
        # The unbiased=False is important to match the original numpy implementation that does not apply Bessel's correction
        # https://pytorch.org/docs/stable/generated/torch.var.html
        batch_var = torch.var(batch, dim=0, unbiased=False)
        batch_size = batch.shape[0]

        # Variance calculation from Wikipedia
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        tot_count = self.count + batch_size
        batch_ratio = batch_size / tot_count

        delta = batch_mean - self.mean
        m_a = self.variance * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta**2 * self.count * batch_ratio
        ## end of variance calculation from Wikipedia ##

        self.mean = self.mean + delta * batch_ratio
        self.variance = M2 / tot_count
        self.count = tot_count

    def normalise(self, batch: torch.Tensor, update=True) -> torch.Tensor:
        if update:
            self.update(batch)
        normalised = (batch - self.mean) / (self.variance + 1e-8) ** 0.5
        return torch.clip(normalised, min=self.clip_min, max=self.clip_max)

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.variance = self.variance.to(device)


def round_col(df: pl.DataFrame, col_name: str, round_value: int):
    col = df[col_name] / round_value
    col = col.round(0).cast(pl.Int64) * round_value
    return df.with_columns(col.alias(col_name))


def stats_by(col_name: str, df: pl.DataFrame):
    # Add an "exit_rate" column that is equal to the "in_elevator" column divided by 4
    if "in_elevator" in df.columns:
        df = df.with_columns((df["in_elevator"] / 4).alias("exit_rate"))
    # df = df.with_columns((df["in_elevator"] / 4).alias("exit_rate"))
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
    # https://www.youtube.com/watch?v=w3tM-PMThXk
    confidence_intervals = list[pl.Series]()
    n_items_per_timestep = grouped.count().sort("time_step")["count"].to_numpy().astype(np.float32)
    for col in cols:
        mean = res[f"mean_{col}"].to_numpy().astype(np.float32)
        # Avoid zero std, otherwise a "inf" * 0 will be computed by scipy, leading to NaN
        std = res[f"std_{col}"].to_numpy().astype(np.float32) + 1e-8
        # Use scipy.stats.t if the sample size is small (then degree of freedom, df, is n_samples - 1)
        # Use scipy.stats.norm if the sample size is large
        lower, upper = sp.norm.interval(0.95, loc=mean, scale=std / n_items_per_timestep**0.5)
        ci95 = (upper - lower) / 2
        new_col = pl.Series(name=f"ci95_{col}", values=ci95)
        confidence_intervals.append(new_col)

    res = res.with_columns(confidence_intervals)
    for series in res.select(pl.col(pl.FLOAT_DTYPES)):
        # Type hinting
        series: pl.Series
        mask = series.is_infinite() | series.is_nan()
        series[mask] = 1e20
        res.replace(series.name, series)
    return res
