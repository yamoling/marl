from marlenv import catalog

import marl

env = catalog.MStepsMatrix(10)

exp = marl.Experiment.load("logs/QMix-LLE-lvl6")
print("loaded")
for r in exp.runs:
    print(r.rundir)
