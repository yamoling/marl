import logging
import os
import sys
from typing import Type

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


def dqn[E: DiscreteMARLEnv](env: EnvConfig[E], mixer: Type[Mixer], rnd: algos.RND | None = None):
    return algos.DQN(
        qnetworks.from_env(env, independent=True),
        TransitionMemory(50_000),
        mixer=mixer.from_env(env),
        train_policy=EpsilonGreedy.linear(1, 0.05, 100_000),
        gamma=0.95,
        train_interval=(5, "step"),
        lr=5e-4,
        batch_size=64,
        optimiser_type="adam",
        grad_norm_clipping=10,
        ir_module=rnd,
    )


def main(args: Args):
    env = LLEConfig(6, obs_type="layered", state_type="layered")
    # env = EnvConfig.from_any(catalog.MStepsMatrix(10), maven_noise_size=16)
    rnd = algos.RND.from_env(env)
    # trainer = maven(env)
    trainer = dqn(env, mixers.VDN, rnd)
    exp = Experiment(env, trainer, logdir="auto", n_steps=1_000_000)
    exp.run(
        seeds=4,
        n_tests=5,
        gpu_strategy="scatter",
        test_interval=5000,
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
