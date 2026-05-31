import logging
import os

import dotenv
import typed_argparse as tap
from lle import CooperationLevel

import marl
from marl.env import LLEPool
from marl.nn.model_bank import actor_critics


class Args(tap.TypedArgs):
    pool_size: int = tap.arg(default=50)


def main(args: Args):
    for size in (100, 250, 500):
        for cooperation in CooperationLevel:
            if cooperation == CooperationLevel.COOPERATIVE:
                continue
            if cooperation == CooperationLevel.FULLY_COUPLED:
                generator = "level6_style"
            else:
                generator = "random"
            env = LLEPool(f"maps/pool/{generator}/{cooperation.name}", size)
            test_env = LLEPool(f"maps/pool/{generator}/{cooperation.name}", size, offset=500)
            actor, critic = actor_critics.from_env(env, False)
            trainer = marl.algos.PPO(
                actor,
                critic,
                marl.nn.mixers.VDN.from_env(env),
                gamma=0.95,
                grad_norm_clipping=10,
                early_stopping_kl=1e-2,
            )
            trainer_name = "IPPO" if trainer.mixer is None else "MAPPO"
            try:
                exp = marl.Experiment.create(
                    env,
                    trainer,
                    test_env=test_env,
                    logdir=f"{trainer_name}-pool-{size}-{cooperation.name}",
                )
                logging.info(f"Created experiment in {exp.logdir}")
            except FileExistsError:
                logging.info(f"Experiment directory already exists for cooperation level {cooperation}. Skipping.")


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("pool.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    tap.Parser(Args).bind(main).run()
