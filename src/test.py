from marl import Experiment, training
from marl.env import EnvConfig, LLEConfig
from marl.models import EpisodeMemory
from marl.nn import mixers
from marl.nn.model_bank import MAVENQnetwork, qnetworks
from marl.policy import EpsilonGreedy


def maven(env: EnvConfig):
    return training.MAVEN(
        MAVENQnetwork.from_env(env),
        EpsilonGreedy.linear(1, 0.01, 100),
        env,
        train_interval=(1, "episode"),
        grad_norm_clipping=10.0,
        batch_size=16,
    )


def vdn(env: EnvConfig):
    return training.VDN(
        qnetworks.from_env(env),
        EpsilonGreedy.linear(1, 0.01, 100),
        EpisodeMemory(5000),
        train_interval=(1, "episode"),
        grad_norm_clipping=10.0,
        batch_size=16,
    )


def qmix(env: EnvConfig):
    return training.QMix(
        qnetworks.from_env(env),
        EpsilonGreedy.linear(1, 0.01, 100),
        EpisodeMemory(5000),
        train_interval=(1, "episode"),
        grad_norm_clipping=10.0,
        batch_size=16,
        mixer=mixers.QMix.from_env(env),
    )


if __name__ == "__main__":
    # env = EnvConfig.from_any(catalog.MStepsMatrix(10), maven_noise_size=16)
    for algo in (vdn, maven, qmix):
        if algo is maven:
            env = LLEConfig(6, maven_noise_size=16)
        else:
            env = LLEConfig(6)
        experiment = Experiment(env, algo(env), logdir="auto", n_steps=2_000_000)
        experiment.run(seeds=8, test_interval=5000, gpu_strategy="scatter", n_tests=5)
