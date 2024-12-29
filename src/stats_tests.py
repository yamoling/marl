from marl.utils import stats
import polars as pl


df = pl.read_csv("/home/yann/projects/python/marl/logs/tests/run_2024-12-29_23:09:14.104930_seed=0/training_data.csv")
print(df)
df = stats.round_col(df, "time_step", 5_000)
# df = df.group_by("time_step").agg(pl.col(col).drop_nulls().mean() for col in df.columns)
print()
