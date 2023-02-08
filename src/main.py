from laser_env import LaserEnv
import rlenv
import marl


def train_n_step(run_num: int) -> list[str]:
    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    n_step_algo = marl.qlearning.NStepReturn(3, memory=marl.models.TransitionSliceMemory(2_000, 3))
    runner = marl.Runner(env, test_env=test_env, algo=n_step_algo)
    runner.seed(run_num)
    return runner.train(n_steps=10_000, test_interval=500, n_tests=5, quiet=True)


def train(run_num: int, n_step: bool) -> str:
    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    dqn = marl.qlearning.DQN(qnetwork, memory=marl.models.TransitionMemory(2_000))
    runner = marl.Runner(env, algo=dqn, test_env=test_env)
    runner.seed(run_num)
    return runner.train(n_steps=10_000, test_interval=500, n_tests=5, quiet=True)


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
    train(0, n_step=True)
    exit()
    run_experiment(n_runs=5, pool_size=7)
    # train_table_qlearning(0)
    # train_vanilla(0)
    # train_vdn_extrinsic(0)
    # train_vdn_plain(0)
