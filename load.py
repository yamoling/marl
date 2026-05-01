import marl


exp = marl.Experiment.load("logs/MAVEN-QMix-return-MStepsMatrix-eps0.01")
results = exp.get_experiment_results()
for label, result in results.items():
    print(f"Label: {label}")
    print(result.collect().head())
