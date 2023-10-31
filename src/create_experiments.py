import marl
import rlenv
from marl.training import DQNTrainer
from marl.training.dqn_trainer import HardUpdate



def create_experiments():
    memory_size = 20_000
    n_steps = 40_000

    env = rlenv.make("CartPole-v1")


    trainer = DQNTrainer(
        qnetwork=marl.nn.model_bank.MLP256.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(1, 0.01, n_steps=20_000),
        memory=marl.models.TransitionMemory(memory_size),
        target_update=HardUpdate(200),
        lr=1e-3,
        update_interval=1
    )

    algo = marl.qlearning.DQN(
        qnetwork=trainer.qnetwork,
        train_policy=trainer.policy
    )


    logdir = f"logs/{env.name}-lr=1e-3-original-trainer"
    logdir = "debug"

    exp = marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=1000, n_steps=n_steps)
    exp.create_runner("csv", seed=0).train(5)


if __name__ == "__main__":
    create_experiments()
    #run(0, 20_000)
