import rlenv
import marl



def main3():
    env = rlenv.make("CartPole-v1")
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    algo = marl.qlearning.RecurrentMixedDQN(qnetwork, mixer)
    logger = marl.logging.CSVLogger("test")
    runner = marl.Runner(env, algo, logger, test_interval=1000, n_steps=25000)
    runner.train(n_tests=3)

def main():
    env, test_env = rlenv.Builder("smac:3m").agent_id().last_action().build_all()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents, 64)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    anneal = 50_000
    decrease_amount = (1 - 0.05) / anneal
    train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps=0.05)
    test_policy = marl.policy.ArgMax()
    memory = marl.models.EpisodeMemory(5_000)
    algo = marl.qlearning.RecurrentMixedDQN(
        qnetwork=qnetwork,
        mixer=mixer,
        lr=5e-4,
        memory=memory,
        train_policy=train_policy,
        test_policy=test_policy,
        batch_size=32
    )
    experiment = marl.Experiment.create("test", algo, env, n_steps=200_000, test_interval=10000, test_env=test_env)
    runner = experiment.create_runner("csv", "tensorboard", quiet=False)
    runner.train(n_tests=50)


if __name__ == "__main__":
    main3()
