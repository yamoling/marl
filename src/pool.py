import logging
import os

import dotenv
import typed_argparse as tap
from lle import CooperationLevel
from lle.solver.cooperation_level import CooperationLevelStr

import marl
from marl.env import LLEPool
from marl.nn.model_bank import qnetworks


class Args(tap.TypedArgs):
    cooperation: CooperationLevelStr = tap.arg(default="independent")
    pool_size: int = tap.arg(default=50)


def main(args: Args):
    for cooperation in CooperationLevel:
        if cooperation == CooperationLevel.COOPERATIVE:
            continue
        if cooperation == CooperationLevel.FULLY_COUPLED:
            generator = "level6_style"
        else:
            generator = "random"
        logging.info(f"Running with cooperation level: {cooperation} and generator: {generator}")
        env = LLEPool(f"maps/pool/{generator}/{cooperation.upper()}", args.pool_size)
        test_env = LLEPool(f"maps/pool/{generator}/{cooperation.upper()}", args.pool_size, offset=500)
        trainer = marl.algos.VDN(
            qnetworks.from_env(env, independent=True, duelling=True),
            gamma=0.95,
            grad_norm_clipping=10.0,
            train_policy=marl.policy.EpsilonGreedy.linear(1.0, 0.025, 200_000),
        )
        exp = marl.Experiment(env, trainer, test_env=test_env, logdir="auto")
        exp.run(
            5,
            n_tests=args.pool_size,
            test_interval=10_000,
            gpu_strategy="scatter",
            disabled_gpus=[0, 1, 2],
        )
        exp.run(
            range(5, 10),
            n_tests=args.pool_size,
            test_interval=10_000,
            gpu_strategy="scatter",
            disabled_gpus=[0, 1, 2],
            save_weights=False,
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("pool.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    tap.Parser(Args).bind(main).run()
