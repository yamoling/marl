from marlenv import catalog

import marl

env = catalog.MStepsMatrix(10)

exp = marl.Experiment.load("logs/VDN-steps")
run = exp.get_run(0)
print(run)
