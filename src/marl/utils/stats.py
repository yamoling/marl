import numpy as np
import polars as pl

class RunningMeanStd(object):
    """Credits to https://github.com/jcwleo/random-network-distillation-pytorch"""
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def round_col(df: pl.DataFrame, col_name: str, round_value: int):
    col = df[col_name] / round_value
    col = col.round(0)
    col = col * round_value
    col = col.cast(pl.Int64)
    return df.with_columns(col.alias(col_name))


def stats_by(col_name: str, df: pl.DataFrame):
    grouped = df.groupby(col_name)
    excluded_cols = ["time_step", "timestamp_sec"]
    return (
        grouped
        .agg(
            [pl.mean(col).alias(f"mean_{col}") for col in df.columns if col not in excluded_cols] +
            [pl.std(col).alias(f"std_{col}") for col in df.columns if col not in excluded_cols] +
            [pl.min(col).alias(f"min_{col}") for col in df.columns if col not in excluded_cols] +
            [pl.max(col).alias(f"max_{col}") for col in df.columns if col not in excluded_cols])
        .sort("time_step")
    )