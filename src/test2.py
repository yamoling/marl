import marl

exp = marl.Experiment.load("logs/pbrs-baseline/")
runs = list(exp.runs)
print(runs)
run = runs[0]
agent = exp.agent
for d in run.test_dirs:
    exp.agent.load(d)
    weights = []
    for network in agent.networks:
        for name, params in network.named_parameters():
            if "weight" in name:
                w = params.detach().numpy()
                print(w.shape)
                weights.append(w)
