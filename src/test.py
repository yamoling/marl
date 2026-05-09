import logging
import os

import dotenv
import lle

import marl
from marl.models import TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.policy import EpsilonGreedy
from marl.training import DQN, PPO


def main():
    env = lle.level(6).obs_type("layered").state_type("state").builder().agent_id().time_limit(78).build()

    # oc = options_nn.CNNOptionCritic.from_env(env, 4)
    # trainer = PPOC(
    #     oc,
    #     env.n_agents,
    #     mixer=mixers.VDN.from_env(env),
    #     option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    #     train_interval=32,
    #     early_stopping_kl=0.01,
    # )
    trainer = DQN(
        qnetworks.QCNN(env.observation_shape, env.extras_size, env.n_actions),
        EpsilonGreedy.linear(1.0, 0.05, 250_000),
        TransitionMemory(50_000),
        mixers.VDN.from_env(env),
        grad_norm_clipping=10.0,
        batch_size=64,
        lr=5e-4,
        gamma=0.95,
    )
    logdir = f"logs/{env.name}-{trainer.name}-old"
    exp = marl.Experiment.create(
        env,
        1_000_000,
        trainer=trainer,
        test_interval=5000,
        logdir=logdir,
        save_weights=False,
        replace_if_exists=True,
    )
    exp.run(seeds=12, n_tests=10, disabled_gpus=[0, 1], fill_strategy="scatter", quiet=True)


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
