from laser_env import LaserEnv
import rlenv
import marl



def train_vanilla(run_id: int):
    map_name = "maps/lvl3"
    log_path = f"logs/vanilla-linear-{map_name}-seed_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .extrinsic_reward("linear", initial_reward=1, anneal=100)\
        .build_all()
    algo = marl.VanillaQLearning(
        env, 
        test_env=test_env, 
        train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-6, min_eps=5e-2),
        test_policy=marl.policy.ArgMax(),
        log_path=log_path
    )
    algo = marl.debugging.FileWrapper(algo, algo.logger.logdir)
    # algo = marl.VanillaQLearning(env, test_env=test_env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5), log_path=log_path)
    algo.seed(run_id)
    algo.train(n_steps=1_000_000, test_interval=5000, n_tests=1)


def train_table_qlearning(run_id: int):
    map_name = "maps/lvl3"
    log_path = f"logs/tabular-replay-linear-{map_name}-seed_{run_id}"
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
        # replay_memory=marl.models.TransitionMemory(50_000),
        replay_memory=marl.models.PrioritizedMemory(marl.models.TransitionMemory(50_000), alpha=0.7, beta=0.4)
    )
    algo = marl.debugging.FileWrapper(algo, algo.logger.logdir)
    algo.seed(run_id)
    algo.train(n_steps=1_000_000, test_interval=5000, n_tests=1)

def train_vdn_extrinsic(run_id: int):
    map_name = "maps/lvl3"
    log_path = f"logs/vdn-extrinsic-linear-{map_name}-seed_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .extrinsic_reward("linear", initial_reward=1, anneal=100)\
        .build_all()
    algo = marl.LinearVDN(
        env, 
        test_env=test_env, 
        train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-6, min_eps=5e-2),
        test_policy=marl.policy.ArgMax(),
        log_path=log_path
    )
    algo = marl.debugging.FileWrapper(algo, algo.logger.logdir)
    algo.seed(run_id)
    algo.train(n_steps=1_000_000, test_interval=5000, n_tests=1)

def train_vdn_plain(run_id: int):
    map_name = "maps/lvl3"
    log_path = f"logs/vdn-plain-{map_name}-seed_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .build_all()
    algo = marl.LinearVDN(
        env, 
        test_env=test_env, 
        train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-6, min_eps=5e-2),
        test_policy=marl.policy.ArgMax(),
        log_path=log_path
    )
    algo = marl.debugging.FileWrapper(algo, algo.logger.logdir)
    algo.seed(run_id)
    algo.train(n_steps=1_000_000, test_interval=5000, n_tests=1)

def train_n_step(run_num: int) -> list[str]:
    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    n_step_algo = marl.qlearning.NStepReturn(
        3,
        env=env,
        test_env=test_env,
        memory=marl.models.TransitionSliceMemory(2_000, 3)
    )
    n_step_algo.seed(run_num)
    return n_step_algo.train(n_steps=10_000, test_interval=500, n_tests=5, quiet=True)


def train_dqn(run_num: int) -> str:
    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    n_step_algo = marl.qlearning.DQN(
        env=env,
        test_env=test_env,
        memory=marl.models.TransitionMemory(2_000)
    )
    n_step_algo.seed(run_num)
    return n_step_algo.train(n_steps=10_000, test_interval=500, n_tests=5, quiet=True)


def run_experiment(n_runs: int, pool_size: int):
    from multiprocessing import Pool
    with Pool(pool_size) as p:
        logdirs = {
            "dqn": p.map(train_dqn, range(n_runs)),
            "n-step dqn": p.map(train_n_step, range(n_runs))
        }
    print(logdirs)


if __name__ == "__main__":
    print(marl.__version__)
    run_experiment(n_runs=5, pool_size=7)
    # train_table_qlearning(0)
    # train_vanilla(0)
    # train_vdn_extrinsic(0)
    # train_vdn_plain(0)
