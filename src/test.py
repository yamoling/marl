import logging
import os

import dotenv
import lle

import marl
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.training import MAVEN


def main():
    NOISE_SIZE = 16
    env = (
        lle.level(6)
        .obs_type("layered")
        .state_type("state")
        .pbrs(gamma=1.0, reward_value=1.0, lasers_to_reward=[(4, 0), (6, 12)])
        .builder()
        .agent_id()
        .time_limit(78)
        .pad("extra", NOISE_SIZE, label="maven")
        .build()
    )
    assert len(env.observation_shape) == 3
    meta_agent_input = (env.observation_shape[0] * env.n_agents, *env.observation_shape[1:])
    trainer = MAVEN(
        qnetworks.MAVENCNN.from_env(env),
        marl.policy.EpsilonGreedy.linear(1.0, 0.05, 50_000),
        "return",
        NOISE_SIZE,
        env.n_actions,
        env.n_agents,
        env.state_size,
        env.state_extras_size,
        return_bandit_nn=qnetworks.QCNN(NOISE_SIZE, meta_agent_input, env.extras_size * env.n_agents),
        mixer=mixers.VDN.from_env(env),
        test_policy=marl.policy.ArgMax(),
        grad_norm_clipping=10.0,
        batch_size=16,
        train_interval=(1, "episode"),
    )
    # trainer = DQN(
    #     qnetworks.QCNN.from_env(env),
    #     marl.policy.EpsilonGreedy.linear(1.0, 0.05, 50_000),
    #     marl.models.EpisodeMemory(5_000),
    #     mixer=mixers.VDN.from_env(env),
    #     test_policy=marl.policy.ArgMax(),
    #     grad_norm_clipping=10.0,
    #     batch_size=16,
    #     train_interval=(1, "episode"),
    # )
    logdir = f"logs/{trainer.name}-{env.name}"
    logdir = "test"
    exp = marl.Experiment.create(
        env,
        1_000_000,
        trainer=trainer,
        test_interval=5000,
        logdir=logdir,
        save_weights=False,
        replace_if_exists=True,
    )
    exp.run(seeds=20, n_tests=10, fill_strategy="scatter", quiet=True, disabled_gpus=[0, 1, 5, 6, 7], n_parallel=1)


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
