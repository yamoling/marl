from laser_env import LaserEnv
import rlenv
import marl


def train(params: tuple[int, bool, bool, bool, str]) -> str:
    run_num, n_step, intrinsic, per, logdir = params
    builder = rlenv.Builder(LaserEnv("maps/lvl3")).agent_id().time_limit(20)
    if run_num == 0:
        builder = builder.record(logdir)
    if intrinsic:
        builder = builder.extrinsic_reward("linear", initial_reward=0.5, anneal=10)
    env, test_env = builder.build_all()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    if n_step:
        memory = marl.models.slice_memory.NStepReturnMemory(10_000, 5)
    else:
        memory = marl.models.TransitionMemory(10_000)
    if per:
        memory = marl.models.PrioritizedMemory(memory, alpha=0.7, beta=0.4)
    algo = marl.qlearning.vdn.VDN(marl.qlearning.DQN(qnetwork, memory=memory))
    if run_num == 0:
        algo = marl.debugging.FileWrapper(algo, logdir)
    runner = marl.Runner(env, algo=algo, test_env=test_env, logdir=logdir)
    runner.seed(run_num)
    return runner.train(n_steps=50_000, test_interval=2500, n_tests=5, quiet=False)


def run_experiment(n_runs: int, pool_size: int):
    from multiprocessing import Pool
    with Pool(pool_size) as p:
        dqn_params = [(i, False, False, False, f"logs/vdn-{i}") for i in range(0)]
        nstep_params =[(i, True, False, False, f"logs/vdn-n_step-{i}") for i in range(n_runs)]
        nstep_per_params =[(i, True, False, True, f"logs/vdn-n_step-{i}") for i in range(n_runs)]
        intrinsic_params = [(i, True, True, False, f"logs/vdn-intrinsic-nstep-{i}") for i in range(n_runs)]
        # dqn_logs = p.map_async(train, dqn_params)
        # nstep_logs = p.map_async(train, nstep_params)
        logdir = p.map_async(train, intrinsic_params)
        print("intrinsic", logdir.get())
        # logdirs = {"dqn": dqn_logs.get(), "nstep": nstep_logs.get()}
    # print(logdirs)


if __name__ == "__main__":
    print(marl.__version__)
    # train((0, True, False, True, "debug"))
    run_experiment(n_runs=5, pool_size=10)
