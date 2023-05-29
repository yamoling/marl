import torch
import polars as pl

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
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / (self.count + batch_count)
        
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
    return (
        grouped
        .agg(
            [pl.mean(col).alias(f"mean_{col}") for col in cols] +
            [pl.std(col).alias(f"std_{col}") for col in cols] +
            [pl.min(col).alias(f"min_{col}") for col in cols] +
            [pl.max(col).alias(f"max_{col}") for col in cols])
        .sort("time_step")
    )