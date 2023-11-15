import marl
import rlenv
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.training.dqn_trainer import HardUpdate, SoftUpdate


def create_experiments():
    memory_size = 50_000
    n_steps = 1_000_000

    env = rlenv.make("CartPole-v1")
    env = rlenv.Builder(LLE.from_file("maps/lvl6-shaping", ObservationType.LAYERED)).agent_id().time_limit(78).build()

    # qnetwork = marl.nn.model_bank.CNN.from_env(env)
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    trainer = DQNTrainer(
        qnetwork,
        train_policy=marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=10_000),
        memory=marl.models.TransitionMemory(memory_size),
        target_update=HardUpdate(200),
        lr=5e-4,
        update_interval=5,
        mixer=marl.qlearning.VDN(env.n_agents),
    )

    algo = marl.qlearning.DQN(qnetwork=qnetwork, train_policy=trainer.policy)

    logdir = "logs/demo"

    exp = marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)
    # runner = exp.create_runner("csv", seed=0)
    # runner.train(10)
    # exp.create_runner("csv", seed=0).train(5)


if __name__ == "__main__":
    create_experiments()
    # run(0, 20_000)
