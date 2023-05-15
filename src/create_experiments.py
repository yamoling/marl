import rlenv
import marl
import os
from laser_env import ObservationType, StaticLaserEnv
from marl.models import Experiment


def create_static_experiments():
    batch_size = 64
    memory_size = 50_000
    levels = ["lvl3", "lvl4", "lvl5", "lvl6"]
    training_steps = [600_000, 1_000_000, 2_000_000, 2_000_000]
    for level, n_steps in zip(levels, training_steps):
        # level = os.path.join("maps/alternating/", level)
        env = StaticLaserEnv(level, ObservationType.LAYERED)
        time_limit = int(env.width * env.height / 2)
        current_env, test_env = rlenv.Builder(env).agent_id().time_limit(time_limit).build_all()
        
        # E-greedy decreasing from 1 to 0.05 over 600000 steps
        min_eps = 0.05
        decrease_amount = (1 - min_eps) / 600_000
        train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
        test_policy = marl.policy.ArgMax()

        qnetwork = marl.nn.model_bank.CNN.from_env(current_env)
        mixer = marl.qlearning.mixers.VDN(env.n_agents)
        # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents, 64)
        memory = marl.models.TransitionMemory(memory_size)
        #memory = marl.models.PrioritizedMemory(memory)

        algo = marl.qlearning.MixedDQN(
            qnetwork=qnetwork,
            batch_size=batch_size,
            train_policy=train_policy,
            test_policy=test_policy,
            gamma=0.95,
            memory=memory,
            mixer=mixer
        )
        
        logdir = f"logs/{level}-{mixer.name}-{memory.__class__.__name__}"
        # logdir = "test"
        exp = Experiment.create(logdir, algo=algo, env=current_env, n_steps=n_steps, test_interval=5000, test_env=test_env)
        print("Created experiment:", exp.logdir)
        # exp.create_runner("csv", quiet=True).train(n_tests=1)
        # exit(0)


if __name__ == "__main__":
    create_static_experiments()
