"""Example script showing JSON-based experiment configuration and recreation."""

from marl.config import (
    DQNTrainerConf,
    EpsilonGreedyPolicyConf,
    ExperimentConf,
    LLEEnvConf,
    NetworkConfig,
    SoftUpdateConf,
    TransitionMemoryConf,
)
from marl.models import Experiment


def build_dqn_conf(logdir: str = "logs/config-example-dqn") -> ExperimentConf:
    env_conf = LLEEnvConf(level=6, obs_type="layered", state_type="layered", agent_id=True, last_action=False)
    env = env_conf.make()

    trainer_conf = DQNTrainerConf(
        qnet=NetworkConfig.from_env(env, mlp_sizes=(128, 128), hidden_activation="relu", noisy=False),
        train_policy=EpsilonGreedyPolicyConf.linear(start=1.0, end=0.05, n_steps=200_000),
        test_policy=EpsilonGreedyPolicyConf.constant(0.0),
        memory=TransitionMemoryConf(max_size=50_000),
        target_updater=SoftUpdateConf(tau=0.01),
        optimiser_type="adam",
        gamma=0.99,
        batch_size=64,
        lr=5e-4,
        train_interval=(5, "step"),
        double_qlearning=True,
        grad_norm_clipping=10.0,
    )

    return ExperimentConf(
        logdir=logdir,
        env=env_conf,
        trainer=trainer_conf,
        n_steps=200_000,
        test_interval=5_000,
        save_weights=True,
        logger="csv",
        save_actions=True,
    )


def main():
    conf = build_dqn_conf()

    # 1) Create experiment directory and write both runtime files and experiment.conf.json
    created = conf.make_experiment(replace_if_exists=True)
    print(f"Created experiment at {created.logdir}")

    # 2) Recreate runtime objects directly from JSON configuration
    loaded = Experiment.load(created.logdir)
    print(f"Reloaded experiment from JSON conf: {loaded.logdir}")
    print(f"Trainer: {loaded.trainer.name}")
    print(f"Environment: {loaded.env.name}")


if __name__ == "__main__":
    main()
