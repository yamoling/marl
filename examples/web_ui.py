import marl
import marlenv
import gymnasium as gym
from lle import LLE, ObservationType
from ui.backend import run as run_server

from gym_example import dqn
from multi_agent_cnn import mappo

LOGDIR1 = "logs/gym-cartpole-dqn"
LOGDIR2 = "logs/lle-level6-mappo"


def run_experiment1():
    # Start 3 runs of the Gym experiment
    N_RUNS = 3
    env = marlenv.adapters.Gym(gym.make("CartPole-v1"))
    algo, trainer = dqn(env)  # type: ignore
    exp = marl.Experiment.create(logdir=LOGDIR1, env=env, agent=algo, trainer=trainer, n_steps=10_000, test_interval=500)
    for seed in range(N_RUNS):
        exp.run(seed)


def run_experiment2():
    # Start a single run of the LLE experiment
    env = env = LLE.level(6).obs_type(ObservationType.LAYERED).build()
    env = marlenv.Builder(env).time_limit(env.width * env.height // 2).agent_id().build()
    algo, trainer = mappo(env)
    exp = marl.Experiment.create(logdir=LOGDIR2, env=env, agent=algo, trainer=trainer, n_steps=10_000, test_interval=1_000)
    exp.run(0)


if __name__ == "__main__":
    try:
        marl.Experiment.load(LOGDIR1)
    except FileNotFoundError:
        run_experiment1()

    try:
        marl.Experiment.load(LOGDIR2)
    except FileNotFoundError:
        run_experiment2()

    run_server()
