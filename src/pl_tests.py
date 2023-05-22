import polars as pl
import marl
from marl.utils import stats


run = marl.Run.load("logs/lvl1-DQN-TransitionMemory/run_1684749550.2095988")
df = run.training_data
print(df)


exp = marl.Experiment.load("logs/lvl1-DQN-TransitionMemory")
print("loaded")
ticks, datasets = exp.training_data()
print(ticks)
print(datasets[0].label)
print(datasets[0].mean)
print(datasets[0].min)
