import logging
import os

import dotenv
import lle

import marl
from marl.nn.model_bank import actor_critics, qnetworks
from marl.nn import mixers
from marl.training import PPO, DQN


def main():
    env = lle.level(6).obs_type("layered").state_type("state").pbrs().builder().agent_id().time_limit(78).build()

    # oc = options_nn.CNNOptionCritic.from_env(env, 4)
    # trainer = PPOC(
    #     oc,
    #     env.n_agents,
    #     mixer=mixers.VDN.from_env(env),
    #     option_train_policy=EpsilonGreedy.linear(1.0, 0.05, 50_000),
    #     train_interval=32,
    #     early_stopping_kl=0.01,
    # )
    trainer = PPO(
        actor_critics.CNNDiscreteAC.from_env(env),
        mixer=None,  # mixers.VDN.from_env(env),
        grad_norm_clipping=10.0,
        early_stopping_kl=0.01,
        n_epochs=15,
        lr_actor=5e-4,
        lr_critic=5e-4,
    )
    trainer = DQN(
        qnetworks.QCNN.from_env(env),
        mixer=mixers.QMix.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(1.0, 0.05, 50_000),
        test_policy=marl.policy.ArgMax(),
        memory=marl.models.TransitionMemory(50_000),
        grad_norm_clipping=10.0,
    )
    logdir = f"logs/{trainer.name}-{env.name}"
    exp = marl.Experiment.create(
        env,
        1_000_000,
        trainer=trainer,
        test_interval=5000,
        logdir=logdir,
        save_weights=False,
        replace_if_exists=True,
    )
    exp.run(seeds=30, n_tests=1, disabled_gpus=[0, 1, 2, 3], fill_strategy="scatter", quiet=True)


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
