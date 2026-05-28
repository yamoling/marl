import logging
import os
import signal
import sys
from typing import Type

import dotenv
import typed_argparse as tap
from marlenv import DiscreteMARLEnv

from marl import Experiment, algos
from marl.env import EnvConfig, LLEConfig
from marl.models import Mixer
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import EpsilonGreedy, SoftmaxPolicy


class Args(tap.TypedArgs):
    _quiet: bool | None = tap.arg("--quiet", default=False)

    @property
    def quiet(self):
        """If the user has explicitly set quiet mode, then use it. Otherwise, if the program is started with nohup, then enable quiet mode."""
        if self._quiet is not None:
            return self._quiet
        # If nohup, then the default SIGHUP signal is not hte default
        if signal.getsignal(signal.SIGHUP) == signal.SIG_DFL:
            return False  # Normal mode
        return True  # Nohup mode


def maven[E: DiscreteMARLEnv](env: EnvConfig[E], rnd: algos.RND | None = None):
    return algos.MAVEN(
        qnetworks.MAVENQnetwork.from_env(env),
        EpsilonGreedy.linear(1, 0.01, 100),
        env,
        ir_module=rnd,
        gamma=0.95,
        lr=5e-4,
        batch_size=16,
        optimiser_type="rmsprop",
        grad_norm_clipping=10,
    )


def dqn[E: DiscreteMARLEnv](
    env: EnvConfig[E],
    mixer: Type[Mixer],
    *,
    rnd: algos.RND | None = None,
    independent: bool = True,
    recurrent: bool = False,
    duelling: bool = True,
    noisy: bool = False,
):
    return algos.DQN(
        qnetworks.from_env(env, recurrent=recurrent, independent=independent, duelling=duelling, noisy=noisy),
        mixer=mixer.from_env(env),
        train_policy=SoftmaxPolicy(5),  # EpsilonGreedy.linear(1, 0.05, 100_000),
        gamma=0.95,
        train_interval=(5, "step"),
        lr=5e-4,
        batch_size=64,
        optimiser_type="adam",
        grad_norm_clipping=10,
        ir_module=rnd,
    )


def main(args: Args):
    env = LLEConfig(6, obs_type="layered", state_type="flattened", maven_noise_size=None)
    trainer = dqn(env, mixers.VDN, rnd=None, independent=False, recurrent=False, duelling=True)
    exp = Experiment(env, trainer, logdir="test", n_steps=1_000_000)
    exp.run(
        seeds=10,
        n_tests=5,
        gpu_strategy="scatter",
        test_interval=5000,
        quiet=args.quiet,
        device="auto",
        n_jobs=1,
    )
    exit(0)
    exp.run(
        seeds=range(10, 20),
        n_tests=5,
        save_weights=False,
        gpu_strategy="scatter",
        test_interval=5000,
        quiet=args.quiet,
        device="auto",
        disabled_gpus=[0, 1, 2, 3, 5, 6, 7],
        n_jobs=5,
    )
    # # env = EnvConfig.from_any(catalog.MStepsMatrix(10), maven_noise_size=16)
    # rnd = algos.RND.from_env(env)
    # # rnd = None
    # # trainer = maven(env, rnd)
    # for duelling in [True, False]:
    #     for independent in [True, False]:
    #         trainer = dqn(env, mixers.VDN, rnd=rnd, independent=independent, recurrent=False, duelling=duelling)
    #         try:
    #             exp = Experiment(env, trainer, logdir="auto", n_steps=1_000_000)
    #             exp.run(
    #                 seeds=12,
    #                 n_tests=5,
    #                 gpu_strategy="scatter",
    #                 test_interval=5000,
    #                 quiet=args.quiet,
    #                 disabled_gpus=[0, 1, 3, 5, 6],
    #                 n_jobs=12,
    #             )
    #         except FileExistsError:
    #             logging.warning("Experiment with logdir 'test' already exists. Skipping this run to avoid overwriting existing results.")


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
