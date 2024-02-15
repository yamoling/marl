from marl import Experiment

res = {}
for run in Experiment.get_runs("logs/StaticLaserEnv-QMix"):
    x = Experiment.compute_datasets([run.train_metrics], True)
    res[run.rundir] = x
