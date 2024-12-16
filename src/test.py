import marl
from lle import LLE
import marlenv
import torch.multiprocessing as mp
from marl.training import DQNTrainer, SoftUpdate


def main():
    env = LLE.from_file("maps/subgraph-2agents-laser.toml").single_objective()
    # env = marlenv.Builder(env).time_limit(env.width * env.height // 2).agent_id().build()

    n_steps = 1_000_000
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
    mixer = marl.algo.VDN.from_env(env)
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
        mixer=mixer,
        grad_norm_clipping=10,
        ir_module=None,
    )

    logdir = "logs/unlimited-time-laser-random"
    local_graph_trainer = marl.algo.intrinsic_reward.IndividualLocalGraphTrainer(env, 5_000, logdir)

    dqn = marl.algo.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    trainer = marl.training.MultiTrainer(local_graph_trainer)  # , dqn_trainer)
    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        algo=dqn,
        trainer=trainer,
        test_interval=test_interval,
        test_env=test_env,
        logdir=logdir,
    )
    exp.run(n_tests=0)


if __name__ == "__main__":
    main()
