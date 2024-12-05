import marl
from lle import LLE
import marlenv
from marl.training import DQNTrainer, SoftUpdate


def main():
    env = LLE.level(3).single_objective()
    env = marlenv.Builder(env).time_limit(78).agent_id().build()

    n_steps = 200_000
    test_interval = 5000
    gamma = 0.95
    test_env = None

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=200_000,
    )
    qmix = marl.algo.VDN.from_env(env)
    dqn_trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=64,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=qmix,
        grad_norm_clipping=10,
        ir_module=None,
    )
    local_graph_trainer = marl.algo.intrinsic_reward.IndividualLocalGraphTrainer(env)

    dqn = marl.algo.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    trainer = marl.training.MultiTrainer(local_graph_trainer, dqn_trainer)
    exp = marl.Experiment.create(env, n_steps, algo=dqn, trainer=trainer, test_interval=test_interval, test_env=test_env)
    exp.run()


if __name__ == "__main__":
    main()
