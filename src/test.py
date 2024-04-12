from run import Arguments, main as run_main
from lle import LLE, ObservationType
import marl
import rlenv
from marl.training import DQNTrainer, SoftUpdate
from marl.env.random_initial_pos import RandomInitialPos

def main(bottleneck_width: int):
    n_steps = 300 #300_000
    test_interval = 5000
    gamma = 0.95
    

    file_path = f"maps/b-width-{bottleneck_width}"
    builder = LLE.from_file(file_path)
    lle = builder.obs_type(ObservationType.LAYERED).build()
    env = lle
    env = RandomInitialPos(env, 0, 1, 0, lle.width - 1)
    # env = BShaping(env, lle.world, 1, args.reward_delay, args.reward_in_laser)
    # env = ZeroPunishment(env)
    env = rlenv.Builder(env).agent_id().time_limit(int(lle.width * lle.height / 1.5), add_extra=True).build()

    # qnetwork = marl.nn.model_bank.CNN.from_env(env, mlp_sizes=(256, 256))
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=50_000,
    )
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=64,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=marl.qlearning.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
        grad_norm_clipping=10,
        # ir_module=rnd,
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    logdir = f"logs/bottleneck-width-{bottleneck_width}"
    exp = marl.Experiment.create(
        logdir,
        algo=algo,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
    )

    args = Arguments(
        logdir=logdir,
        n_runs=2,
        n_tests=10,
    )
    run_main(args)


if __name__ == "__main__":
    for width in range(1, 5):
        main(width)