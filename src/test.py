from analysis import aggregate
import rlenv
import marl

def train(run_id: int, per: bool):
    log_path = f"logs/cartpole-seed-{run_id}"
    memory = marl.models.TransitionMemory(10_000)
    if per:
        log_path = f"{log_path}-per"
        memory = marl.models.PrioritizedMemory(memory, alpha=0.7, beta=0.4)
        memory._tree.seed(run_id)
    else:
        log_path = f"{log_path}-plain"
    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    
    
    algo = marl.LinearVDN(env, test_env, log_path=log_path)
    # algo = marl.ReplayTableQLearning(
    #     env, 
    #     test_env=test_env, 
    #     train_policy=marl.policy.DecreasingEpsilonGreedy(env.n_agents, decrease_amount=5e-3, min_eps=5e-2),
    #     test_policy=marl.policy.ArgMax(),
    #     log_path=log_path,
    #     replay_memory=memory,
    #     batch_size=32
    # )
    algo.seed(run_id)
    algo.train(n_steps=10_000, test_interval=500, n_tests=10)


if __name__ == "__main__":
    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    n_step_algo = marl.qlearning.NStepReturn(
        3,
        env=env,
        test_env=test_env,
        memory=marl.models.TransitionSliceMemory(2_000, 3)
    )

    env, test_env = rlenv.Builder("CartPole-v1").build_all()
    dqn = marl.qlearning.DQN(
        env=env,
        test_env=test_env,
        memory=marl.models.TransitionMemory(2_000)
    )
    
    logdirs = {
        "3-step return": [n_step_algo.train(n_steps=10_000, test_interval=1000)],
        "plain": [dqn.train(n_steps=10_000, test_interval=1000)]
    }
    print(logdirs)
    aggregate.aggregate(logdirs, "3-step comparison")