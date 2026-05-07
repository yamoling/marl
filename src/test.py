import logging
import os

import dotenv

from marl import Experiment, training
from marl.env import EnvConfig, LLEConfig
from marl.models import EpisodeMemory, TransitionMemory
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
        TransitionMemory(50000),
        train_interval=(5, "step"),
        grad_norm_clipping=10.0,
        batch_size=64,
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
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("test.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    env = LLEConfig(6)
    experiment = Experiment(env, vdn(env), logdir="VDN-steps", n_steps=1_000_000)
    experiment.run(seeds=8, test_interval=5000, gpu_strategy="scatter", n_tests=5)
