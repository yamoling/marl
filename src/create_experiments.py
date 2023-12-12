import marl
import rlenv
from rlenv.wrappers import RLEnvWrapper
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.training.qtarget_updater import SoftUpdate


def create_smac(map_name="8m"):
    n_steps = 1_000_000
    env = rlenv.adapters.SMAC(map_name)
    env = rlenv.Builder(env).agent_id().build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=500_000)
    test_policy = marl.policy.EpsilonGreedy.constant(0.05)
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        optimiser="adam",
        batch_size=8,
        update_interval=1,
        gamma=0.99,
        train_every="episode",
        # mixer=marl.qlearning.VDN(env.n_agents),
        mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
        grad_norm_clipping=10,
    )

    algo = marl.qlearning.RDQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=test_policy,
    )
    logdir = f"logs/smac-{map_name}-qmix"
    logdir = "logs/test"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_lle():
    n_steps = 1_000_000
    env = rlenv.Builder(LLE.level(6, ObservationType.LAYERED)).agent_id().time_limit(78, add_extra=True).build()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=500_000)
    test_policy = marl.policy.EpsilonGreedy.constant(0.05)
    # memory = marl.models.PrioritizedMemory(
    #     memory=memory,
    #     alpha=0.6,
    #     beta=marl.utils.Schedule.linear(0.4, 1.0, n_steps),
    #     priority_clipping=5.0,
    # )
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        optimiser="adam",
        batch_size=64,
        update_interval=5,
        gamma=0.95,
        train_every="step",
        mixer=marl.qlearning.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
        grad_norm_clipping=10,
        ir_module=marl.intrinsic_reward.RandomNetworkDistillation(
            obs_shape=env.observation_shape,
            extras_shape=env.extra_feature_shape,
        ),
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=test_policy,
    )

    logdir = f"logs/{env.name}-vdn-rnd"
    # logdir = "logs/test"

    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


if __name__ == "__main__":
    exp = create_smac()
    # exp = create_lle()
    runner = exp.create_runner(seed=0)
    runner.to("cuda")
    runner.train(0)
