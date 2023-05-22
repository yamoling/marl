import rlenv
import marl
from laser_env import StaticLaserEnv, ObservationType




def main():
    env, test_env = rlenv.Builder(StaticLaserEnv("maps/normal/lvl6", ObservationType.LAYERED)).agent_id().time_limit(50).build_all()
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 100_000)
    test_policy = marl.policy.ArgMax()
    ir = marl.intrinsic_reward.RandomNetworkDistillation(env.observation_shape, env.extra_feature_shape)
    algo = marl.qlearning.LinearMixedDQN(
        qnetwork=qnetwork,
        mixer=mixer,
        lr=5e-4,
        train_policy=train_policy,
        test_policy=test_policy,
        batch_size=32,
        ir_module=ir,
        train_interval=5
    )
    experiment = marl.Experiment.create("test", algo, env, n_steps=200_000, test_interval=10000, test_env=test_env)
    runner = experiment.create_runner("csv", quiet=False)
    runner.train(n_tests=1)


if __name__ == "__main__":
    main()
