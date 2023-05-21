import torch
import rlenv
import marl



def main():
    marl.seed(1)
    env, test_env = rlenv.Builder("smac:3m").agent_id().build_all()
    env.seed(1)
    test_env.seed(1)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 50_000)
    test_policy = marl.policy.ArgMax()
    memory = marl.models.EpisodeMemory(5000)
    optimizer = torch.optim.RMSprop(list(qnetwork.parameters()) + list(mixer.parameters()), lr=5e-4, alpha=0.99, eps=1e-5)

    algo = marl.qlearning.RecurrentMixedDQN(
        qnetwork=qnetwork,
        mixer=mixer,
        gamma=0.99,
        batch_size=32,
        double_qlearning=True,
        train_policy=train_policy,
        optimizer=optimizer,
        memory=memory,
        test_policy=test_policy,
    )

    logdir = "smac:3m-qmix"
    logdir = "test"
    exp = marl.Experiment.create(logdir, algo, env, test_env=test_env, test_interval=10_000, n_steps=200_000)
    runner = exp.create_runner(quiet=False)
    runner._algo.logger = runner._logger
    runner.train(32)


if __name__ == "__main__":
    main()