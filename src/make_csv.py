import marl



exp = marl.Experiment.load("logs/smac-vdn")
ticks, datasets = exp.test_metrics()
for ds in datasets:
    with open(f"{ds.label}.csv", "w") as f:
        f.write("x,mean, plus_std, minus_std\n")
        for tick, mean, std, mmin, mmax in zip(ticks, ds.mean, ds.std, ds.min, ds.max):
            f.write(f"{tick},{mean},{min(mean + std, mmax)},{max(mean - std, mmin)}\n")