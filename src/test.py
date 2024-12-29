from typing import Any
import marl
from lle import LLE
import marlenv
from marlenv import DiscreteActionSpace
from marl.training import DQNTrainer, SoftUpdate
from marl.training.continuous_ppo_trainer import ContinuousPPOTrainer
from marl.training.haven_trainer import HavenTrainer
from marl.agents import VDN
from marl.nn.model_bank.actor_critics import CNNContinuousActorCritic


def make_vdn_agent(env: marlenv.MARLEnv[Any, DiscreteActionSpace], gamma: float):
    return DQNTrainer(
        qnetwork=marl.nn.model_bank.qnetworks.CNN.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(
            1.0,
            0.05,
            n_steps=200_000,
        ),
        memory=marl.models.TransitionMemory(50_000),
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=marl.agents.VDN.from_env(env),
        grad_norm_clipping=10.0,
    )


def main_ppo_haven():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    N_SUBGOALS = 16
    K = 4

    lle = LLE.level(6).obs_type("layered").build()
    width = lle.width
    height = lle.height
    meta_env = marlenv.Builder(lle).time_limit(width * height // 2).build()
    meta_ppo = ContinuousPPOTrainer(
        actor_critic=CNNContinuousActorCritic(
            input_shape=meta_env.observation_shape,
            n_extras=meta_env.extra_shape[0],
            action_output_shape=(
                meta_env.n_agents,
                N_SUBGOALS,
            ),
        ),
        batch_size=1024,
        minibatch_size=64,
        n_epochs=32,
        value_mixer=VDN.from_env(meta_env),
        gamma=gamma,
        lr=5e-4,
    )
    env = marlenv.Builder(meta_env).agent_id().pad("extra", N_SUBGOALS).build()
    dqn_trainer = make_vdn_agent(env, gamma)

    meta_trainer = HavenTrainer(
        meta_trainer=meta_ppo,
        worker_trainer=dqn_trainer,
        k=K,
        n_subgoals=N_SUBGOALS,
        n_meta_extras=meta_env.extra_shape[0],
        n_agent_extras=env.extra_shape[0] - meta_env.extra_shape[0] - N_SUBGOALS,
    )

    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        trainer=meta_trainer,
        test_interval=test_interval,
        test_env=None,
        logdir="logs/tests",
    )
    exp.run()


def main_dqn_haven():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    N_SUBGOALS = 16
    K = 4

    lle = LLE.level(6).obs_type("layered").build()
    width = lle.width
    height = lle.height
    meta_env = marlenv.Builder(lle).time_limit(width * height // 2).build()
    meta_ppo = ContinuousPPOTrainer(
        actor_critic=CNNContinuousActorCritic(
            input_shape=meta_env.observation_shape,
            n_extras=meta_env.extra_shape[0],
            action_output_shape=(
                meta_env.n_agents,
                N_SUBGOALS,
            ),
        ),
        batch_size=256,
        minibatch_size=64,
        value_mixer=VDN.from_env(meta_env),
        gamma=gamma,
        lr=5e-4,
    )
    env = marlenv.Builder(meta_env).agent_id().pad("extra", N_SUBGOALS).build()

    dqn_trainer = DQNTrainer(
        qnetwork=marl.nn.model_bank.qnetworks.CNN.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(
            1.0,
            0.05,
            n_steps=200_000,
        ),
        memory=marl.models.TransitionMemory(50_000),
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=marl.agents.VDN.from_env(env),
        grad_norm_clipping=10,
    )

    meta_trainer = HavenTrainer(
        meta_trainer=meta_ppo,
        worker_trainer=dqn_trainer,
        k=K,
        n_subgoals=N_SUBGOALS,
        n_meta_extras=meta_env.extra_shape[0],
        n_agent_extras=env.extra_shape[0] - meta_env.extra_shape[0] - N_SUBGOALS,
    )

    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        trainer=meta_trainer,
        test_interval=test_interval,
        test_env=None,
        logdir="logs/tests",
    )
    exp.run()


def main_vdn():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95

    lle = LLE.level(6).obs_type("layered").build()
    width = lle.width
    height = lle.height
    env = marlenv.Builder(lle).time_limit(width * height // 2).agent_id().build()

    dqn_trainer = make_vdn_agent(env, gamma)
    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        trainer=dqn_trainer,
        test_interval=test_interval,
        test_env=None,
        logdir="logs/tests-vdn",
    )
    exp.run()


def main_lunar_lander_continuous():
    import gymnasium as gym
    from marlenv.adapters import Gym

    env = Gym(gym.make("LunarLanderContinuous-v3", render_mode="rgb_array"))
    actor_critic = marl.nn.model_bank.actor_critics.MLPContinuousActorCritic(
        input_shape=env.observation_shape,
        n_extras=env.extra_shape[0],
        action_output_shape=(env.n_actions,),
    )
    trainer = ContinuousPPOTrainer(
        actor_critic=actor_critic,
        value_mixer=VDN.from_env(env),  # type: ignore
        gamma=0.99,
        lr=1e-3,
        exploration_c2=0.01,
    )
    exp = marl.Experiment.create(
        env,
        1_000_000,
        trainer=trainer,
        test_interval=5_000,
        # logdir="logs/lunar-lander-ppo-with-entropy",
    )
    exp.run(
        render_tests=False,
        n_tests=5,
    )


if __name__ == "__main__":
    main_ppo_haven()
    # main_vdn()
    # main_lunar_lander_continuous()
