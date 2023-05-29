import polars as pl
import marl
from marl.utils import stats



exp = marl.Experiment.load("logs/lvl6-VDN-TransitionMemory-RND-p0.25")
print("loaded")
ticks, datasets = exp.train_metrics()

