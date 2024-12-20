import marl
from lle import LLE
import marlenv
from marl.agents import Haven
from marl.training import DQNTrainer, SoftUpdate


def main():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    N_SUBGOALS = 16
    K = 4

    lle = LLE.level(6).obs_type("layered").state_type("layered").single_objective().build()
    width = lle.width
    height = lle.height
    meta_env = marlenv.Builder(lle).time_limit(width * height // 2).build()
    meta_network = marl.nn.model_bank.CNN(
        input_shape=meta_env.observation_shape,
        extras_size=meta_env.extra_shape[0],
        output_shape=(
            meta_env.n_agents,
            N_SUBGOALS,
        ),
    )

    env = marlenv.Builder(meta_env).agent_id().pad("extra", N_SUBGOALS).build()
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

    dqn = marl.agents.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )
    meta_agent = Haven(meta_network, dqn, N_SUBGOALS, K)

    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        agent=meta_agent,
        trainer=dqn_trainer,
        test_interval=test_interval,
        test_env=None,
        logdir=logdir,
    )
    exp.run(n_tests=1)


if __name__ == "__main__":
    main()
