import logging
import os

import dotenv
import lle

import marl
from marl.nn import mixers
from marl.nn.model_bank import actor_critics
from marl.nn.model_bank import options as options_nn
from marl.policy import EpsilonGreedy
from marl.training import PPO, PPOC


def main():
    env = lle.level(6).obs_type("layered").state_type("state").builder().agent_id().time_limit(78).build()

    oc = options_nn.CNNOptionCritic.from_env(env, 4)
    trainer = PPOC(
        oc,
        env.n_agents,
        mixer=marl.nn.mixers.VDN.from_env(env),
        option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
        train_interval=32,
    )
    trainer = PPO(actor_critics.CNN_ActorCritic.from_env(env), mixer=mixers.VDN.from_env(env), early_stopping_kl=0.015)
    logdir = f"logs/{env.name}-{trainer.name}"
    logdir = "test"
    exp = marl.Experiment.create(env, 1_000_000, trainer=trainer, test_interval=5000, logdir=logdir, save_weights=True)
    exp.run(seeds=[0], n_tests=5, disabled_gpus=[1, 2, 3], n_parallel=16, fill_strategy="scatter")


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("test.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred during execution.", exc_info=e)
