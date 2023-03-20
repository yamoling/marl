import rlenv
import marl
from laser_env import ObservationType, DynamicLaserEnv



if __name__ == "__main__":
    logdir = "logs/dynamic-flattened-softmax"
    env, test_env = rlenv.Builder(DynamicLaserEnv(
        width=5,
        height=5,
        num_agents=2,
        obs_type=ObservationType.FLATTENED,
        num_gems=5,
        wall_density=0.15
    )).agent_id().time_limit(30).build_all()
    
    algo = (marl.DeepQBuilder()
            .vdn()
            .train_policy(marl.policy.SoftmaxPolicy(env.n_actions))
            .qnetwork_default(env)
            .build())
    algo = marl.wrappers.ReplayWrapper(algo, logdir)
    logger = marl.logging.TensorBoardLogger(logdir)
    runner = marl.Runner(env, algo, logger, test_env=test_env)
    runner.train(2_000_000, test_interval=5000, quiet=True)
    

