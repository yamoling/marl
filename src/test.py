from typing import Literal
from lle import LLE
import marlenv
from marl.training import DQNTrainer, SoftUpdate
from marl.training.intrinsic_reward import AdvantageIntrinsicReward
from marl.training.continuous_ppo_trainer import ContinuousPPOTrainer
from marl.training.haven_trainer import HavenTrainer
from marl.training.mixers import VDN
from marl.nn.model_bank.actor_critics import CNNContinuousActorCritic
from marl.utils import MultiSchedule, Schedule

import marl


def make_haven(agent_type: Literal["dqn", "ppo"], ir: bool):
    n_steps = 2_000_000
    test_interval = 5000
    WARMUP_STEPS = 1
    gamma = 0.95
    N_SUBGOALS = 16
    K = 4

    lle = LLE.level(6).obs_type("layered").state_type("layered").build()
    width = lle.width
    height = lle.height
    meta_env = marlenv.Builder(lle).time_limit(width * height // 2).agent_id().build()

    if ir:
        value_network = marl.nn.model_bank.actor_critics.CNNCritic(meta_env.state_shape, meta_env.state_extra_shape[0])
        ir_module = AdvantageIntrinsicReward(value_network, gamma)
    else:
        ir_module = None

    match agent_type:
        case "ppo":
            meta_agent = ContinuousPPOTrainer(
                actor_critic=CNNContinuousActorCritic(
                    input_shape=meta_env.observation_shape,
                    n_extras=meta_env.extras_shape[0],
                    action_output_shape=(N_SUBGOALS,),
                ),
                batch_size=1024,
                minibatch_size=64,
                n_epochs=32,
                value_mixer=VDN.from_env(meta_env),
                gamma=gamma,
                lr=5e-4,
                # grad_norm_clipping=10.0,
            )
        case "dqn":
            meta_agent = DQNTrainer(
                qnetwork=marl.nn.model_bank.qnetworks.CNN(
                    input_shape=meta_env.observation_shape,
                    extras_size=meta_env.extras_shape[0],
                    output_shape=(N_SUBGOALS,),
                ),
                train_policy=marl.policy.EpsilonGreedy.linear(1.0, 0.05, 200_000),
                memory=marl.models.TransitionMemory(5_000),
                double_qlearning=True,
                target_updater=SoftUpdate(0.01),
                lr=5e-4,
                train_interval=(1, "step"),
                use_ir=False,
                train_ir=True,
                ir_module=ir_module,
                gamma=gamma,
                mixer=VDN.from_env(meta_env),
                grad_norm_clipping=10.0,
            )
        case other:
            raise ValueError(f"Invalid agent type: {other}")

    env = marlenv.Builder(meta_env).pad("extra", N_SUBGOALS).build()
    worker_trainer = DQNTrainer(
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
        mixer=VDN.from_env(env),
        grad_norm_clipping=10.0,
        ir_module=ir_module,
        train_ir=False,
        use_ir=True,
    )

    meta_trainer = HavenTrainer(
        meta_trainer=meta_agent,
        worker_trainer=worker_trainer,
        n_subgoals=N_SUBGOALS,
        n_workers=env.n_agents,
        k=K,
        n_meta_extras=meta_env.extras_shape[0],
        n_agent_extras=env.extras_shape[0] - meta_env.extras_shape[0] - N_SUBGOALS,
        n_meta_warmup_steps=WARMUP_STEPS,
    )

    return marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        trainer=meta_trainer,
        test_interval=test_interval,
        test_env=None,
        # logdir=f"logs/haven-{agent_type}{'-ir' if ir else ''}-clipping=10",
    )
    # exp.run()


def main_lunar_lander_continuous():
    import gymnasium as gym
    from marlenv.adapters import Gym

    env = Gym(gym.make("LunarLanderContinuous-v3", render_mode="rgb_array"))
    actor_critic = marl.nn.model_bank.actor_critics.MLPContinuousActorCritic(
        input_shape=env.observation_shape,
        n_extras=env.extras_shape[0],
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


def main_cartpole_vdn():
    import gymnasium as gym
    from marlenv.adapters import Gym
    from marl.policy import EpsilonGreedy

    env = Gym(gym.make("CartPole-v1", render_mode="rgb_array"))

    trainer = DQNTrainer(
        marl.nn.model_bank.MLP.from_env(env),
        EpsilonGreedy.linear(1.0, 0.05, 10_000),
        marl.models.TransitionMemory(10_000),
        mixer=VDN.from_env(env),  # type: ignore
        lr=1e-3,
    )
    exp = marl.Experiment.create(
        env,
        1_000_000,
        trainer=trainer,
        test_interval=2500,
        logdir="logs/debug",
    )
    exp.run(
        render_tests=False,
        n_tests=5,
    )


if __name__ == "__main__":
    # main_cartpole_vdn()
    # make_haven("dqn", True)
    # make_haven("ppo", True).run()
    make_haven(agent_type="dqn", ir=True).run()
    # make_haven("ppo", False)
    # main_vdn()
    # main_lunar_lander_continuous()
