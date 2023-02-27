import rlenv
import marl
from laser_env import LaserEnv


if __name__ == "__main__":
    gamma = 0.99
    n_steps = 5
    memory_size = 5000
    logdir = "test"

    env, test_env = rlenv.Builder(LaserEnv("maps/lvl3")).time_limit(20).agent_id().build_all()
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    memory = marl.models.MemoryBuilder(memory_size, "transition").nstep(n_steps, gamma).build()
    algo = marl.VDN(marl.DQN(qnetwork=qnetwork, gamma=gamma, memory=memory))
    
    runner = marl.Runner(env, algo, test_env=test_env, logdir=logdir)
    runner.train(n_steps=10000, test_interval=1000, n_tests=2)
