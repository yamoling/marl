import marl
import rlenv
from lle import LLE, ObservationType
from marl import Experiment
from marl.training import DQNTrainer


def create_experiments():
    memory_size = 50_000
    level = "lvl6"
    n_steps = 1_000_000

    env = LLE.from_file(level, ObservationType.LAYERED)
    time_limit = round(env.width * env.height / 2)
    env, test_env = rlenv.Builder(env).agent_id().time_limit(time_limit).build_all()

    # E-greedy decreasing from 1 to 0.05 over 100_000 update steps
    train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, 100_000)
    test_policy = marl.policy.ArgMax()


    # ir = marl.intrinsic_reward.RandomNetworkDistillation(
    #     env.observation_shape,
    #     env.extra_feature_shape,
    #     update_ratio=0.25,
    #     clip_value=1,
    #     ir_weight=Schedule.linear(2.0, 0, 300_000)
    # )
    trainer = DQNTrainer(
        qnetwork=marl.nn.model_bank.CNN.from_env(env),
        memory=marl.models.TransitionMemory(memory_size),
        gamma=0.95,
        mixer=marl.qlearning.mixers.VDN(env.n_agents)
    )

    algo = marl.qlearning.DQN(
        trainer=trainer,
        train_policy=train_policy,
        test_policy=test_policy,
    )

    logdir = f"logs/{level}-VDN"
    # logdir = "test"
    exp = Experiment.create(logdir, algo=algo, env=env, test_interval=5000, test_env=test_env, n_steps=n_steps)
    # exp.create_runner().train(1)
    print("Created experiment:", exp.logdir)


if __name__ == "__main__":
    create_experiments()
