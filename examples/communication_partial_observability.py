import marl
from lle import LLE, ObservationType
import marlenv
from marl.agents.qlearning.maic import MAICParameters


def main():
    env = LLE.level(3).obs_type(ObservationType.PARTIAL_7x7).state_type(ObservationType.FLATTENED).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    parameters = MAICParameters(n_agents=env.n_agents)

    maic_network = marl.nn.model_bank.qnetworks.MAICNetworkCNN.from_env(env, parameters)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        50_000,
    )

    algo = marl.agents.MAIC(
        maic_network=maic_network,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
        args=parameters,
    )
    batch_size = 32
    # Add the MAICTrainer (MAICLearner)
    trainer = marl.training.MAICTrainer(
        args=parameters,
        maic_network=maic_network,
        train_policy=train_policy,
        batch_size=batch_size,
        memory=marl.models.EpisodeMemory(5000),
        gamma=0.95,
        mixer=marl.training.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents), #TODO: try with QMix : state needed
        double_qlearning=True,
        target_updater=marl.training.SoftUpdate(0.01),
        lr=5e-4,
        grad_norm_clipping=10,
    )
    runner = marl.Runner(env, algo, trainer)
    runner.run("logs/example-communication", n_steps=10_000, test_interval=1000, n_tests=10)


if __name__ == "__main__":
    main()
