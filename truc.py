import marl

exp = marl.Experiment.load("logs/VDN-steps")
run = exp.get_run(0)
print(run)
