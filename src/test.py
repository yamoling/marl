import rlenv
import marl
from marl.models import Experiment
from laser_env import ObservationType, DynamicLaserEnv


if __name__ == "__main__":
    logdir = "logs/pid-tests"
    env, test_env = rlenv.Builder(DynamicLaserEnv(
        width=5,
        height=5,
        num_agents=2,
        obs_type=ObservationType.FLATTENED,
        num_gems=5,
        wall_density=0.15
    )).agent_id().time_limit(30).build_all()
    
    anneal = 200_000
    eps_start = 1.
    eps_min = 0.1
    eps_decrease = (eps_start - eps_min) / anneal
    algo = (marl.DeepQBuilder()
            .vdn()
            .gamma(0.95)
            .train_policy(marl.policy.DecreasingEpsilonGreedy(eps_start, eps_decrease, eps_min))
            .qnetwork_default(env)
            .build())
    experiment = Experiment.create(logdir, algo=algo, env=env)
    runner = experiment.create_runner("both", seed=0)
    runner.train(3_000_000, test_interval=5000, quiet=True, n_tests=5)
