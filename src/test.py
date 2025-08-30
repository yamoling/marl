import marl
import marlenv

if __name__ == "__main__":
    env = marlenv.catalog.DeepSea(25)
    env = marl.env.StateCounter(env)

    exp = marl.Experiment.create(env, 100_000, logger="neptune")
    # exp.run(seed=seed, n_tests=20)
