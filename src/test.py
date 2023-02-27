import rlenv
import marl
from laser_env import LaserEnv


if __name__ == "__main__":
    gamma = 0.99
    n_steps = 5
    memory_size = 20_000
    n_steps = 1_000_000
    n_tests = 1
    logdir = "logs/intrinsic_reward_2000"

    env, test_env = (rlenv.Builder(LaserEnv("maps/lvl3"))
                     .intrinsic_reward("linear", 1., 2000)
                     .time_limit(20)
                     .agent_id()
                     .add_logger("action", logdir)
                     .build_all())
    memory = (marl.models.MemoryBuilder(memory_size, "transition")
              .nstep(n_steps, gamma)
              .prioritized(alpha=1, eps=1e-3)
              .build())
    algo = (marl.DeepQBuilder(env=env)
            .vdn()
            .memory(memory)
            .gamma(gamma)
            .build())
    
    runner = marl.Runner(env, algo, test_env=test_env, logdir=logdir)
    runner.train(n_steps=n_steps, test_interval=1000, n_tests=n_tests)
