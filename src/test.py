import rlenv
import marl
from laser_env import Difficulty, ObservationType
from marl.utils.env_pool import pool_from_zip
from marl.models import Experiment



if __name__ == "__main__":
    map_size = 5
    difficulty = Difficulty.HARD
    time_limit = map_size * map_size
    current_env, test_env = pool_from_zip(f"maps/{map_size}x{map_size}.zip", difficulty, ObservationType.FLATTENED)
    
    current_env = rlenv.Builder(current_env).agent_id().time_limit(time_limit).build()
    test_env = rlenv.Builder(test_env).agent_id().time_limit(time_limit).build()
    
    
    # E-greedy decreasing from 1 to 0.05 over 600000 steps
    min_eps = 0.05
    decrease_amount = (1 - min_eps) / 600_000
    algo = marl.utils.RandomAgent(current_env.n_actions, current_env.n_agents)
    train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
    test_policy = marl.policy.ArgMax()
    qnetwork = marl.nn.model_bank.MLP.from_env(current_env)
    algo = (marl.DeepQBuilder()
            .qnetwork(qnetwork)
            .train_policy(train_policy)
            .test_policy(test_policy)
            .gamma(0.95)
            .build())

    logdir = f"logs/{algo.name}-{map_size}x{map_size}-{difficulty.name}"
    # logdir = "test"

    experiment = Experiment.create(logdir, algo=algo, env=current_env, n_steps=1_000_000, test_interval=5000, test_env=test_env)
