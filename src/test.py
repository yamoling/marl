import logging
import os
import sys

import dotenv
import typed_argparse as tap
from marlenv import DiscreteMARLEnv, catalog

from marl import Experiment, algos
from marl.env import EnvConfig, LLEConfig
from marl.models import Mixer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import EpsilonGreedy


class Args(tap.TypedArgs):
    quiet: bool = tap.arg(default=False)


def maven[E: DiscreteMARLEnv](env: EnvConfig[E]):
    return algos.MAVEN(
        qnetworks.MAVENQnetwork.from_env(env),
        EpsilonGreedy.linear(1, 0.01, 100),
        env,
        gamma=0.99,
        lr=5e-4,
        batch_size=16,
        optimiser_type="rmsprop",
        grad_norm_clipping=10,
    )


def dqn[E: DiscreteMARLEnv](env: EnvConfig[E], mixer: Mixer):
    return algos.VDN(
        qnetworks.from_env(env, independent=True),
        TransitionMemory(50_000),
        train_policy=EpsilonGreedy.linear(1, 0.05, 100_000),
        gamma=0.95,
        train_interval=(5, "step"),
        lr=5e-4,
        batch_size=64,
        optimiser_type="adam",
        grad_norm_clipping=10,
    )


def main(args: Args):
    # env = LLEConfig(6, obs_type="layered", maven_noise_size=16)
    env = EnvConfig.from_any(catalog.MStepsMatrix(10), maven_noise_size=16)
    trainer = maven(env)
    exp = Experiment(env, trainer, logdir="MAVEN-bmm-return-MStepsMatrix-200k", n_steps=200_000)
    exp.run(
        seeds=4,
        n_tests=5,
        gpu_strategy="scatter",
        test_interval=2000,
        quiet=args.quiet,
        n_jobs=2,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("test.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except Exception as e:
        logging.error(f"An error occurred while starting a run with command line '{sys.argv}'.\nError: {e}", exc_info=True)
