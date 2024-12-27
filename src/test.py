import marl
from lle import LLE
import marlenv
from marl.training import DQNTrainer, SoftUpdate
from marl.training.continuous_ppo_trainer import ContinuousPPOTrainer
from marl.training.haven_trainer import HavenTrainer
from marl.agents import VDN


def main():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    N_SUBGOALS = 16
    K = 4

    lle = LLE.level(6).obs_type("layered").state_type("layered").single_objective().build()
    width = lle.width
    height = lle.height
    env = marlenv.Builder(lle).time_limit(width * height // 2).agent_id().pad("extra", N_SUBGOALS).build()
    # Improvements: do not give the agent ID and the paddings to the meta network
    meta_network = marl.nn.model_bank.actor_critics.CNNContinuousActorCritic(
        input_shape=env.observation_shape,
        n_extras=env.extra_shape[0],
        action_output_shape=(
            env.n_agents,
            N_SUBGOALS,
        ),
    )
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=200_000,
    )
    dqn_trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=marl.models.TransitionMemory(50_000),  # type: ignore
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=64,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=marl.agents.VDN.from_env(env),
        grad_norm_clipping=10,
        ir_module=None,
    )

    logdir = "logs/tests"
    meta_trainer = HavenTrainer(
        worker_trainer=dqn_trainer,
        actor_critic=meta_network,
        gamma=gamma,
        n_epochs=20,
        lr=1e-4,
        eps_clip=0.2,
        k=K,
        c1=1.0,
        exploration_c2=0.01,
    )

    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        trainer=meta_trainer,
        test_interval=test_interval,
        test_env=None,
        logdir=logdir,
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
        logdir="logs/lunar-lander-ppo-with-entropy",
    )
    exit()
    exp.run(
        render_tests=False,
        n_tests=5,
    )


if __name__ == "__main__":
    # main()
    main_lunar_lander_continuous()
