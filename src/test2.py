import marl


def main():
    exp = marl.LightExperiment.load("logs/Overcooked-VDN")
    results = exp.get_experiment_results(replace_inf=True)
    print(results)


if __name__ == "__main__":
    main()
