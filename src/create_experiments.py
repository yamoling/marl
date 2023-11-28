import marl
import rlenv
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.training.qtarget_updater import SoftUpdate


def create_experiments():
    n_steps = 1_000_000

    env = rlenv.Builder(LLE.level(6, ObservationType.LAYERED)).agent_id().time_limit(78, add_extra=True).build()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    # memory = marl.models.PrioritizedMemory(
    #     memory=memory,
    #     alpha=0.6,
    #     beta=marl.utils.Schedule.linear(0.4, 1.0, n_steps),
    # )
    trainer = DQNTrainer(
        qnetwork,
        train_policy=marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=500_000),
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
        grad_norm_clipping=10,
        # ir_module=marl.intrinsic_reward.RandomNetworkDistillation(
        #    obs_shape=env.observation_shape,
        #    extras_shape=env.extra_feature_shape,
        # ),
    )

    algo = marl.qlearning.DQN(qnetwork=qnetwork, train_policy=trainer.policy)

    logdir = f"logs/{env.name}-vdn-double"
    # logdir = "logs/test"

    exp = marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)
    # runner = exp.create_runner(seed=0)
    # runner.to("cuda")
    # runner.train(5)


if __name__ == "__main__":
    create_experiments()
    # run(0, 20_000)
