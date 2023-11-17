import marl
import rlenv
from lle import LLE, ObservationType
from marl.training import DQNTrainer
from marl.training.dqn_trainer import HardUpdate, SoftUpdate


def create_experiments():
    n_steps = 1_000_000

    env = rlenv.Builder(rlenv.adapters.SMAC("3m")).agent_id().build()

    # qnetwork = marl.nn.model_bank.CNN.from_env(env)
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    trainer = DQNTrainer(
        qnetwork,
        train_policy=marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000),
        memory=marl.models.EpisodeMemory(5000),
        target_update=HardUpdate(200),
        lr=5e-4,
        optimiser="rmsprop",
        batch_size=32,
        update_interval=1,
        train_every="episode",
        mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
    )

    algo = marl.qlearning.RDQN(qnetwork=qnetwork, train_policy=trainer.policy)

    logdir = f"logs/{env.name}-qmix"
    # logdir = "logs/test"

    exp = marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)
    # runner = exp.create_runner("csv", seed=0)
    # runner.to("cuda")
    # runner.train(1)


if __name__ == "__main__":
    create_experiments()
    # run(0, 20_000)
