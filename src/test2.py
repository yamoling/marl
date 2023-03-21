import rlenv
import marl
from laser_env import ObservationType, DynamicLaserEnv



if __name__ == "__main__":
    logdir = "logs/test"
    env, test_env = (rlenv.Builder(DynamicLaserEnv(
        width=5,
        height=5,
        num_agents=1,
        obs_type=ObservationType.FLATTENED,
        num_gems=5,
        wall_density=0.15
    ))
        .agent_id()
        .time_limit(30)
        # .pad("extra", 3)
        .penalty(0.1)
        .build_all())
    
    algo = (marl.DeepQBuilder()
            .vdn()
            .train_policy(marl.policy.DecreasingEpsilonGreedy(1, 1e-5, 0.1))
            .qnetwork_default(env)
            .build())
    # algo.load("logs/progressive-egreedy/test/95000")
    algo = marl.wrappers.ReplayWrapper(algo, logdir)
    logger = marl.logging.TensorBoardLogger(logdir)
    runner = marl.Runner(env, algo, logger, test_env=test_env)
    runner.train(10_000, test_interval=1000, quiet=True)
    

