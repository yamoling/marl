from laser_env import LaserEnv
import rlenv
import marl


def train2(run_id: int=0):
    map_name = "maps/lvl3"
    log_path = f"logs/VDN-extrinsic_exponential-{map_name}-run_{run_id}"
    env, test_env = rlenv.Builder(LaserEnv(map_name))\
        .agent_id()\
        .record(f"{log_path}/videos")\
        .extrinsic_reward("exp", anneal=50, initial_reward=1)\
        .build_all()
    
    algo = marl.RecurrentVDN(env, test_env=test_env, train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-5), log_path=log_path)
    algo.seed(run_id)
    algo.train(n_steps=1000, test_interval=2_500, n_tests=1)



def run_experiment(n_runs: int):
    from multiprocessing import Pool
    with Pool(4) as p:
        # p.map(train, range(n_runs))
        p.map(train2, range(n_runs))
        # p.map(train3, range(n_runs))


if __name__ == "__main__":
    print(marl.__version__)
    #run_experiment(n_runs=5)
    train2()
