from laser_env import LaserEnv
import rlenv
import marl


def train(run_num: int, n_step: bool, logdir: str) -> str:
    builder = rlenv.Builder(LaserEnv("maps/lvl3")).agent_id().time_limit(20)
    if run_num == 0:
        builder = builder.record(logdir)
    env, test_env = builder.build_all()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    if n_step:
        memory = marl.models.TransitionSliceMemory(50_000, 5)
    else:
        memory = marl.models.TransitionMemory(50_000)
    algo = marl.qlearning.vdn.VDN(marl.qlearning.DQN(qnetwork, memory=memory))
    if run_num == 0:
        algo = marl.debugging.FileWrapper(algo, logdir)
    if n_step:
        algo = marl.qlearning.NStepReturn(algo, 5)
    runner = marl.Runner(env, algo=algo, test_env=test_env, logdir=logdir)
    runner.seed(run_num)
    return runner.train(n_steps=10_000, test_interval=500, n_tests=5)


def run_experiment(n_runs: int, pool_size: int):
    from multiprocessing import Pool
    with Pool(pool_size) as p:
        dqn_logs = p.map_async(lambda i: train(i, False, f"logs/vdn-{i}"), range(n_runs))
        nstep_logs = p.map_async(lambda i: train(i, True, f"logs/vdn-n_step-{i}"), range(n_runs))
        logdirs = {"dqn": dqn_logs.get(), "nstep": nstep_logs.get()}
    print(logdirs)


if __name__ == "__main__":
    print(marl.__version__)
    train(0, n_step=False, logdir="logs/debug")
    exit()
    run_experiment(n_runs=5, pool_size=7)
