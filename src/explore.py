import rlenv
from laser_env import ObservationType, DynamicLaserEnv
import marl


if __name__ == '__main__':
    env = rlenv.Builder(DynamicLaserEnv(
        width=5,
        height=5,
        num_agents=2,
        obs_type=ObservationType.FLATTENED,
        num_gems=5,
        wall_density=0.15
    )).agent_id().time_limit(30).build()
    #env = rlenv.make("CartPole-v1")
    policy_network = marl.nn.model_bank.MLP.from_env(env)
    algo = marl.policy_gradient.Reinforce(0.99, policy_network)
    experiment = marl.Experiment.create("reinforce-dynamic-lasers", algo, env)
    runner = experiment.create_runner("both", quiet=True)
    runner.train(100_000, n_tests=5, test_interval=2000)
