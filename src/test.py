import logging
import os

import dotenv
import lle
from marlenv import Builder, catalog

import marl
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl import training

NOISE_SIZE = 16


def make_lle():
    return (
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


def make_nsteps_matrix(with_padding: bool):
    builder = Builder(catalog.MStepsMatrix(10)).agent_id()
    if with_padding:
        builder = builder.pad("extra", NOISE_SIZE, label="maven")
    return builder.build()


def main():
    use_maven = True
    env = make_nsteps_matrix(with_padding=use_maven)
    # assert len(env.observation_shape) == 3
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.01, 100)
    if use_maven:
        meta_agent_input = env.observation_shape[0] * env.n_agents
        trainer = training.MAVEN(
            qnetworks.MAVENMLP.from_env(env),
            train_policy,
            NOISE_SIZE,
            env.n_actions,
            env.n_agents,
            env.state_size,
            env.state_extras_size,
            z_policy_type="return",
            return_bandit_nn=qnetworks.QMLP((NOISE_SIZE,), meta_agent_input, (env.extras_size - NOISE_SIZE) * env.n_agents),
            mixer=mixers.QMix.from_env(env, maven_noise_size=NOISE_SIZE),
            test_policy=marl.policy.ArgMax(),
            grad_norm_clipping=10.0,
            batch_size=32,
            train_interval=(1, "episode"),
        )
    else:
        trainer = training.DQN(
            qnetworks.QMLP.from_env(env),
            train_policy,
            marl.models.EpisodeMemory(5000),
            mixer=mixers.QMix.from_env(env),
            test_policy=marl.policy.ArgMax(),
            grad_norm_clipping=10.0,
            batch_size=32,
            train_interval=(1, "episode"),
        )
    logdir = f"logs/{trainer.name}-{env.name}-eps0.01"
    logdir = "test"
    exp = marl.Experiment.create(
        env,
        100_000,
        trainer=trainer,
        test_interval=2000,
        logdir=logdir,
        save_weights=False,
    )
    exp.run(seeds=20, n_tests=10, fill_strategy="scatter", quiet=True, disabled_gpus=[0, 1, 2], n_parallel=1)


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
