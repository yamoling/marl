import logging
import os
import sys
from pathlib import Path
from typing import Literal

import dotenv
import typed_argparse as tap
from marlenv import DiscreteMARLEnv

import marl
from marl import Trainer
from marl.algos import DQN, PPO, VDN, QMix
from marl.env import EnvConfig, LLEPool
from marl.nn import mixers, model_bank


class Args(tap.TypedArgs):
    quiet: bool | None = tap.arg("--quiet", default=False)
    pool_size: int = tap.arg("--pool-size")
    algo: Literal["dqn", "vdn", "qmix", "mappo", "ippo"] = tap.arg("--algo")
    cooperative: bool = tap.arg("--cooperative")
    grid_size: int = tap.arg("--grid-size")
    n_agents: int = tap.arg("--n-agents")
    n_lasers: int = tap.arg("--n-lasers")
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    n_seeds: int = tap.arg("--n-seeds", default=10)

    @property
    def _setting_dir(self):
        return f"{self.grid_size}x{self.grid_size}_agents{self.n_agents}_lasers{self.n_lasers}"

    @property
    def pool_dir(self):
        return Path("maps") / self._setting_dir / ("cooperative" if self.cooperative else "independent")

    @property
    def logdir(self):
        return Path(
            "logs", f"{self.pool_size}-{self._setting_dir}_{self.algo}_{'coop' if self.cooperative else 'indep'}"
        )

    def _make_dqn(self, env: EnvConfig[DiscreteMARLEnv]):
        qnetwork = model_bank.qnetworks.from_env(env, independent=True)
        match self.algo:
            case "dqn":
                return DQN(qnetwork)
            case "vdn":
                return VDN(qnetwork)
            case "qmix":
                return QMix(qnetwork, mixer=mixers.QMix.from_env(env))
        raise ValueError(f"Unknown algorithm: {self.algo}")

    def trainer(self, env: EnvConfig[DiscreteMARLEnv]) -> Trainer:
        if self.algo in ("dqn", "qmix", "vdn"):
            return self._make_dqn(env)

        actor, critic = model_bank.actor_critics.from_env(env, False, independent=True)
        if self.algo == "mappo":
            mixer = mixers.VDN.from_env(env)
        else:
            mixer = None
        return PPO(actor, critic, mixer=mixer)


def main(args: Args):
    train_env = LLEPool(
        args.pool_dir,
        args.pool_size,
        time_limit=args.grid_size**2,
    )
    test_env = LLEPool(
        args.pool_dir,
        args.pool_size,
        offset=500,
        time_limit=args.grid_size**2,
    )
    trainer = args.trainer(train_env)
    exp = marl.Experiment.create(train_env, trainer, test_env=test_env, logdir=args.logdir, n_steps=args.n_steps)
    # exp.run(args.n_seeds, gpu_strategy="scatter", disabled_gpus=range(6), n_jobs=8, n_tests=args.pool_size)
    logging.info(f"Created experiment in {exp.logdir}")


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
        logging.error(
            f"An error occurred while starting a run with command line '{sys.argv}'.\nError: {e}", exc_info=True
        )
