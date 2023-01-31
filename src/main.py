from laser_env import LaserEnv
import rlenv
import marl



def train(run_id: int=0):
    map_name = "maps/lvl3"
    log_path = f"logs/tabular-linear-{map_name}-seed_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .extrinsic_reward("linear", initial_reward=1, anneal=100)\
        .build_all()
    algo = marl.ReplayTableQLearning(
        env, 
        test_env=test_env, 
        train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-6, min_eps=5e-2),
        test_policy=marl.policy.ArgMax(),
        log_path=log_path,
        replay_memory=marl.models.TransitionMemory(50_000)
    )
    # algo = marl.VanillaQLearning(env, test_env=test_env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5), log_path=log_path)
    algo.seed(run_id)
    algo.train(n_steps=1_000_000, test_interval=5000, n_tests=1)

def train2(run_id: int=0):
    map_name = "maps/lvl3"
    log_path = f"logs/tabular-linear-{map_name}-run_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .extrinsic_reward("linear", initial_reward=1., anneal=50)\
        .build_all()
    
    algo = marl.TableQLearning(env, test_env=test_env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5), log_path=log_path)
    algo.seed(run_id)
    algo.train(n_steps=100_000, test_interval=2_500, n_tests=10)

def train3(run_id: int=0):
    map_name = "maps/lvl3"
    log_path = f"logs/tabular-exponential-{map_name}-run_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .extrinsic_reward("exp", initial_reward=1., anneal=50)\
        .build_all()
    
    algo = marl.TableQLearning(env, test_env=test_env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5), log_path=log_path)
    algo.seed(run_id)
    algo.train(n_steps=100_000, test_interval=2_500, n_tests=10)

def run_experiment(n_runs: int, pool_size: int):
    from multiprocessing import Pool
    with Pool(pool_size) as p:
        p.map(train, range(n_runs))
        p.map(train2, range(n_runs))
        p.map(train3, range(n_runs))


if __name__ == "__main__":
    print(marl.__version__)
    # run_experiment(n_runs=5, pool_size=10)
    train(run_id=1)
