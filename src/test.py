from lle import LLE
import marl
import marlenv


def main():
    env = marlenv.make("CartPole-v1")
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 20_000)
    dqn = marl.algo.DQN(qnetwork, policy)
    trainer = marl.training.DQNTrainer(
        qnetwork,
        policy,
        train_interval=(1, "step"),
        memory=marl.models.TransitionMemory(5000),
    )
    exp = marl.Experiment.create(env, 20_000, algo=dqn, trainer=trainer, test_interval=1000)
    exp.run(n_tests=5)


if __name__ == "__main__":
    main()
