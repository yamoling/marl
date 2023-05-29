import rlenv
import marl
from laser_env import ObservationType, StaticLaserEnv
from marl import Experiment
from marl.utils.schedule import Schedule


def create_experiments():
    batch_size = 64
    memory_size = 50_000
    gamma = 0.95
    level = "lvl6"
    level = f"maps/normal/{level}"
    n_steps = 1_000_000
    for n in range(3, 11, 2):
        env = StaticLaserEnv(level, ObservationType.LAYERED)
        time_limit = round(env.width * env.height / 2)
        env, test_env = rlenv.Builder(env).agent_id().time_limit(time_limit).build_all()
        
        # E-greedy decreasing from 1 to 0.05 over 100_000 update steps
        train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 100_000)
        # train_policy = marl.policy.EpsilonGreedy.constant(0.05)
        test_policy = marl.policy.ArgMax()

        qnetwork = marl.nn.model_bank.CNN.from_env(env)
        mixer = marl.qlearning.mixers.VDN(env.n_agents)
        # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents)
        # memory = marl.models.TransitionMemory(memory_size)
        memory = marl.models.NStepMemory(memory_size, n, gamma)

        ir = marl.intrinsic_reward.RandomNetworkDistillation(
            env.observation_shape, 
            env.extra_feature_shape, 
            update_ratio=0.25,
            clip_value=1,
            ir_weight=Schedule.linear(2.0, 0, 300_000)
        )

        algo = marl.qlearning.LinearMixedDQN(
            qnetwork=qnetwork,
            batch_size=batch_size,
            train_policy=train_policy,
            test_policy=test_policy,
            gamma=gamma,
            memory=memory,
            mixer=mixer,
            train_interval=5,
            ir_module=ir
        )

        name = level
        if level.startswith("maps/"):
            name = level[-4:]
        logdir = f"logs/{name}-{mixer.name}-{memory.name}-{memory._n}"
        #logdir = "test"
        exp = Experiment.create(logdir, algo=algo, env=env, test_interval=5000, test_env=test_env, n_steps=n_steps)
        # exp.create_runner().train(1)
        print("Created experiment:", exp.logdir)




if __name__ == "__main__":
    create_experiments()
