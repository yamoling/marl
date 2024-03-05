import marl
import lle
import rlenv
import typed_argparse as tap
from marl.training import DQNNodeTrainer
from marl.training.ppo_trainer import PPOTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate


class Arguments(tap.TypedArgs):
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")
    n_tests: int = tap.arg(default=0, help="Number of tests to run")

class Args:
    def __init__(self):
        self.latent_dim = 64
        self.hidden_size = 128
        self.rnn_hidden_dim = 256
        self.attention_dim = 64
        self.var_floor = 1e-6

def create_smac(args: Arguments):
    n_steps = 2_000_000
    env = rlenv.adapters.SMAC("3s_vs_5z")
    smac = env._env
    env = rlenv.Builder(env).agent_id().last_action().build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(5_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000)
    test_policy = train_policy
    smac_unit_state_size: int = 4 + smac.shield_bits_ally + smac.unit_type_bits
    trainer = DQNNodeTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        double_qlearning=True,
        target_updater=HardUpdate(200),
        lr=5e-4,
        optimiser="adam",
        batch_size=32,
        train_interval=(1, "episode"),
        gamma=0.99,
        mixer=marl.qlearning.Qatten(
            env.n_agents,
            env.n_actions,
            env.state_shape[0],
            smac_unit_state_size,
            weighted_head=False,
        ),
        grad_norm_clipping=10,
    )

    algo = marl.qlearning.RDQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=test_policy,
    )
    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/{env.name}"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}-weighted-head"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_ppo_lle():
    n_steps = 300_000
    env = lle.LLE.level(2, lle.ObservationType.LAYERED)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
    memory = marl.models.TransitionMemory(20)

    trainer = PPOTrainer(
        network=ac_network,
        memory=memory,
        gamma=0.99,
        batch_size=5,
        lr_critic=1e-4,
        lr_actor=1e-4,
        optimiser="adam",
        train_every="step",
        update_interval=20,
        clip_eps=0.2,
        c1=1,
        c2=0.01,
    )

    algo = marl.policy_gradient.PPO(ac_network=ac_network)
    logdir = f"logs/{env.name}-TEST-PPO"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_lle(args: Arguments):
    n_steps = 1_000_000
    gamma = 0.95
    env = lle.LLE.level(6, lle.ObservationType.LAYERED, state_type=lle.ObservationType.FLATTENED)
    env = rlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=False).build()

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=500_000,
    )
    # rnd = marl.intrinsic_reward.RandomNetworkDistillation(
    #     target=marl.nn.model_bank.CNN(env.observation_shape, env.extra_feature_shape[0], 512),
    #     normalise_rewards=False,
    #     # gamma=gamma,
    # )
    rnd = None
    # memory = marl.models.PrioritizedMemory(
    #     memory=memory,
    #     alpha=0.6,
    #     beta=marl.utils.Schedule.linear(0.4, 1.0, n_steps),
    #     td_error_clipping=5.0,
    # )
    trainer = DQNNodeTrainer(
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
        ir_module=None,
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/new-qnetworks-{env.name}"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)

def create_lle_maic(args: Arguments):
    n_steps = 1_000_000
    gamma = 0.95
    env = lle.LLE.level(6, lle.ObservationType.LAYERED, state_type=lle.ObservationType.FLATTENED)
    env = rlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=False).build()

    maic_args = Args()
    qnetwork = marl.nn.model_bank.MAICNetwork.from_env(env, maic_args)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=500_000,
    )
    # rnd = marl.intrinsic_reward.RandomNetworkDistillation(
    #     target=marl.nn.model_bank.CNN(env.observation_shape, env.extra_feature_shape[0], 512),
    #     normalise_rewards=False,
    #     # gamma=gamma,
    # )
    rnd = None
    # memory = marl.models.PrioritizedMemory(
    #     memory=memory,
    #     alpha=0.6,
    #     beta=marl.utils.Schedule.linear(0.4, 1.0, n_steps),
    #     td_error_clipping=5.0,
    # )
    trainer = DQNNodeTrainer(
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
        ir_module=None,
    )

    algo = marl.qlearning.RDQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/new-maic_network-{env.name}"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def main(args: Arguments):
    # exp = create_smac(args)
    # exp = create_ppo_lle()
    exp = create_lle_maic(args)
    #exp = create_lle(args)
    print(exp.logdir)
    if args.run:
        exp.create_runner(seed=0).to("auto").train(args.n_tests)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()