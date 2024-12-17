import marl
from lle import LLE
from lle.tiles import Direction
import marlenv
from marl.training import DQNTrainer, SoftUpdate
from marl.env.wrappers.potential_shaping import LLEPotentialShaping


def main():
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95

    env = LLE.level(6).obs_type("layered").state_type("layered").single_objective()
    l1 = env.world.laser_sources[4, 0]
    l2 = env.world.laser_sources[6, 12]
    width = env.width
    height = env.height
    env = LLEPotentialShaping(env, {l1: Direction.SOUTH, l2: Direction.SOUTH}, gamma)
    env = marlenv.Builder(env).time_limit(width * height // 2).agent_id().build()

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

    logdir = "logs/test"

    dqn = marl.agents.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    exp = marl.Experiment.create(
        env=env,
        n_steps=n_steps,
        agent=dqn,
        trainer=dqn_trainer,
        test_interval=test_interval,
        test_env=None,
        logdir=logdir,
    )
    exp.run(n_tests=1)


if __name__ == "__main__":
    main()
