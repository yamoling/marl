import marl
import rlenv
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.utils import Schedule


def create_experiments():
    memory_size = 50_000
    n_steps = 1_000_000

    #env = LLE.level(6, ObservationType.LAYERED)
    env = rlenv.make("CartPole-v1")
    #time_limit = round(env.width * env.height / 2)
    #env = rlenv.Builder(env).agent_id().time_limit(time_limit).build()

    # E-greedy decreasing from 1 to 0.05 over 100_000 update steps
    train_policy = marl.policy.EpsilonGreedy.linear(1, 0.05, n_steps=10_000)
    test_policy = marl.policy.ArgMax()


    #ir = marl.intrinsic_reward.RandomNetworkDistillation(
    #    env.observation_shape,
    #    env.extra_feature_shape,
    #    update_ratio=0.25,
    #    clip_value=1,
    #    ir_weight=Schedule.constant(1.0)
    #)
    trainer = DQNTrainer(
        #qnetwork=marl.nn.model_bank.CNN.from_env(env),
        qnetwork=marl.nn.model_bank.MLP.from_env(env),
        memory=marl.models.TransitionMemory(memory_size),
        gamma=0.95,
        lr=1e-4,
        double_qlearning=True,
        # ir_module=ir,
        mixer=marl.qlearning.mixers.VDN(env.n_agents)
    )

    algo = marl.qlearning.DQN(
        trainer=trainer,
        train_policy=train_policy,
        test_policy=test_policy,
    )

    logdir = f"logs/{env.name}-VDN"
    logdir = "test"
    exp = marl.Experiment.create(logdir, algo=algo, env=env, test_interval=500, n_steps=n_steps)
    exp.create_runner().train(5)
    print("Created experiment:", exp.logdir)


if __name__ == "__main__":
    create_experiments()
