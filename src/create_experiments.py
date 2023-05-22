import rlenv
import marl
from laser_env import ObservationType, StaticLaserEnv
from marl.models import Experiment


def create_experiments():
    batch_size = 64
    memory_size = 50_000
    level = "lvl6"
    level = f"maps/normal/{level}"
    n_steps = 1_000_000
    env = StaticLaserEnv(level, ObservationType.LAYERED)
    time_limit = round(env.width * env.height / 2)
    env, test_env = rlenv.Builder(env).agent_id().time_limit(time_limit).build_all()
    
    # E-greedy decreasing from 1 to 0.05 over 100_000 update steps
    train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 100_000)
    test_policy = marl.policy.ArgMax()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents)
    memory = marl.models.TransitionMemory(memory_size)
    ir = marl.intrinsic_reward.RandomNetworkDistillation(env.observation_shape, env.extra_feature_shape, update_ratio=0.25)

    algo = marl.qlearning.LinearMixedDQN(
        qnetwork=qnetwork,
        batch_size=batch_size,
        train_policy=train_policy,
        test_policy=test_policy,
        gamma=0.95,
        memory=memory,
        mixer=mixer,
        train_interval=5,
        ir_module=ir
    )

    name = level
    if level.startswith("maps/"):
        name = level[-4:]
    logdir = f"logs/{name}-{mixer.name}-{memory.__class__.__name__}-RND-p0.25"
    #logdir = "test"
    exp = Experiment.create(logdir, algo=algo, env=env, test_interval=5000, test_env=test_env, n_steps=n_steps)
    # exp.create_runner().train(1)
    print("Created experiment:", exp.logdir)


def create_smac_experiments():
    marl.seed(0)
    batch_size = 32
    memory_size = 50_000
    n_steps = 200_000
    env, test_env = rlenv.Builder("smac:3m").agent_id().last_action().build_all()
    
    # E-greedy decreasing from 1 to 0.05 over 400_000 steps
    train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 50_000)
    test_policy = marl.policy.ArgMax()

    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    mixer = marl.qlearning.mixers.VDN(env.n_agents)
    # mixer = marl.qlearning.mixers.QMix(env.state_shape[0], env.n_agents, 64)
    memory = marl.models.EpisodeMemory(memory_size)

    algo = marl.qlearning.RecurrentMixedDQN(
        qnetwork=qnetwork,
        batch_size=batch_size,
        train_policy=train_policy,
        test_policy=test_policy,
        gamma=0.99,
        memory=memory,
        mixer=mixer,
    )

    logdir = f"logs/smac-vdn"
    # logdir = "test-mixed-vdn-double-qlearning"
    exp = Experiment.create(logdir, algo=algo, env=env, n_steps=n_steps, test_interval=5000, test_env=test_env)
    # exp.create_runner().train(1)
    print("Created experiment:", exp.logdir)


if __name__ == "__main__":
    create_experiments()
    # create_smac_experiments()
