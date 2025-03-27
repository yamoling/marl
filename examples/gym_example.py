import marlenv
import gymnasium as gym
import marl
from marlenv import DiscreteActionSpace


def ppo[A](env: marlenv.MARLEnv[A, DiscreteActionSpace]) -> tuple[marl.Agent, marl.Trainer]:
    nn = marl.nn.model_bank.SimpleActorCritic.from_env(env)
    train_policy = marl.policy.CategoricalPolicy()
    algo = marl.agents.PPO(
        ac_network=nn,
        train_policy=train_policy,
    )
    trainer = marl.training.PPO(
        network=nn,
        memory=marl.models.TransitionMemory(20),
    )
    return algo, trainer


def dqn[A](env: marlenv.MARLEnv[A, DiscreteActionSpace]) -> tuple[marl.Agent, marl.Trainer]:
    qnetwork = marl.nn.model_bank.MLP.from_env(env)
    train_policy = marl.policy.EpsilonGreedy.constant(0.1)

    algo = marl.agents.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
    )
    trainer = marl.training.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        memory=marl.models.TransitionMemory(5_000),
        mixer=marl.training.VDN.from_env(env),
        train_interval=(1, "step"),
        lr=5e-4,
    )
    return algo, trainer


if __name__ == "__main__":
    env = marlenv.adapters.Gym(gym.make("CartPole-v1", render_mode="rgb_array"))
    algo, trainer = ppo(env)  # type: ignore
    runner = marl.Runner(env, algo, trainer)
    runner.run(logdir="logs/debug", n_steps=10_000, test_interval=1000, n_tests=10)
