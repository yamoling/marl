import rlenv
import marl
from marl.models import Experiment


def create_cartpole():
    batch_size = 64
    memory_size = 10_000
    env = rlenv.make("CartPole-v1")
    test_env = rlenv.make("CartPole-v1")
    marl.seed(0)

    min_eps = 0.01
    decrease_amount = 2e-4
    train_policy = marl.policy.DecreasingEpsilonGreedy(1, decrease_amount, min_eps)
    test_policy = marl.policy.ArgMax()

    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    memory = marl.models.TransitionMemory(memory_size)
    # memory = marl.models.PrioritizedMemory(memory)
    
    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        batch_size=batch_size,
        train_policy=train_policy,
        test_policy=test_policy,
        gamma=0.99,
        memory=memory,
    )

    logdir = f"logs/{env.name}-{memory.__class__.__name__}-{algo.name}"
    logdir = "cartpole-dqn"
    print("Creating experiment:", logdir)
    exp = Experiment.create(logdir, algo=algo, env=env, n_steps=20_000, test_interval=500, test_env=test_env)
    runner = exp.create_runner("csv")
    runner.train(n_tests=3)



if __name__ == "__main__":
    # create_static_experiments()
    create_cartpole()
    # exp = marl.Experiment.load("logs/LunarLander-PrioritizedMemory-DQN")
    # runner = exp.create_runner("csv")
    # runner.train(n_tests=3)
