import logging
import os

import dotenv

from marl import Experiment, training
from marl.env import LLEConfig
from marl.models import EpisodeMemory, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import MAVENQnetwork, qnetworks
from marl.policy import EpsilonGreedy

if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("test.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for algo in ("vdn",):
        for pbrs in (False, True):
            policy = EpsilonGreedy.linear(1, 0.05, 100_000)
            memory = TransitionMemory(50_000)
            train_interval = (5, "step")
            batch_size = 64
            gamma = 0.95
            lr = 5e-4
            grad_norm_clipping = 10.0
            optimiser_type = "adam"
            match algo:
                case "vdn":
                    env = LLEConfig(6, pbrs=pbrs)
                    trainer = training.VDN(
                        qnetworks.from_env(LLEConfig(6)),
                        memory,
                        train_policy=policy,
                        gamma=gamma,
                        train_interval=train_interval,
                        lr=lr,
                    )
                case "qmix":
                    env = LLEConfig(6, pbrs=pbrs)
                    trainer = training.QMix(
                        qnetworks.from_env(env),
                        memory,
                        mixers.QMix.from_env(env),
                        train_policy=policy,
                        gamma=gamma,
                        train_interval=train_interval,
                        lr=lr,
                        grad_norm_clipping=grad_norm_clipping,
                        optimiser_type=optimiser_type,
                    )
                case "maven":
                    env = LLEConfig(6, maven_noise_size=16, pbrs=pbrs)
                    trainer = training.MAVEN(
                        MAVENQnetwork.from_env(env),
                        policy,
                        env,
                        gamma=gamma,
                        batch_size=16,
                        train_interval=(1, "episode"),
                        lr=lr,
                        grad_norm_clipping=grad_norm_clipping,
                        optimiser_type=optimiser_type,
                        memory=EpisodeMemory(5000),
                    )
            logdir = f"logs/{algo}-{pbrs}-{env.name}"
            experiment = Experiment(env, trainer, logdir=logdir, n_steps=1_000_000)
            experiment.run(seeds=8, test_interval=5000, gpu_strategy="scatter", n_tests=1, n_jobs=8, disabled_gpus=range(4))
