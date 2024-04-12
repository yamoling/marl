import shutil
import marl
import rlenv
from typing import Optional
import typed_argparse as tap
from marl.training import DQNTrainer, DDPGTrainer, PPOTrainer, CNetTrainer, MAICTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.utils import ExperimentAlreadyExistsException
from lle import WorldState, LLE, ObservationType
from run import Arguments as RunArguments, main as run_experiment
from types import SimpleNamespace


class Arguments(RunArguments):
    map_file: str = tap.arg(help="The map file to use")
    reward_in_laser: bool = tap.arg(default=False, help="Whether the reward is given in the laser or not")
    reward_delay: int = tap.arg(help="The number of steps before the reward is given")
    logdir: Optional[str] = tap.arg(default=None, help="The experiment directory")
    overwrite: bool = tap.arg(default=False, help="Override the existing experiment directory")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")


def create_smac(args: Arguments):
    n_steps = 2_000_000
    env = rlenv.adapters.SMAC("3s_vs_5z")
    env = rlenv.Builder(env).agent_id().last_action().build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.EpisodeMemory(5_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000)
    test_policy = train_policy
    trainer = DQNTrainer(
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
        mixer=marl.qlearning.mixers.QPlex(
            n_agents=env.n_agents,
            n_actions=env.n_actions,
            state_size=env.state_shape[0],
            adv_hypernet_embed=64,
            n_heads=10,
            weighted_head=True,
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
            logdir += f"-{trainer.mixer.name}-validation"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_ddpg_lle(args: Arguments):
    n_steps = 500_000
    env = lle.LLE.level(2, obs_type=lle.ObservationType.LAYERED, state_type=lle.ObservationType.FLATTENED)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.DDPG_NN_TEST.from_env(env)
    memory = marl.models.TransitionMemory(5_000)

    train_policy = marl.policy.CategoricalPolicy()
    test_policy = marl.policy.ArgMax()

    trainer = DDPGTrainer(
        network=ac_network, memory=memory, batch_size=64, optimiser="adam", train_every="step", update_interval=5, gamma=0.95
    )

    algo = marl.policy_gradient.DDPG(ac_network=ac_network, train_policy=train_policy, test_policy=test_policy)
    logdir = f"logs/{env.name}-TEST-DDPG"
    if args.debug:
        logdir = "logs/debug"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_ppo_lle(args: Arguments):
    n_steps = 500_000
    env = lle.LLE.level(3, lle.ObservationType.LAYERED)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
    memory = marl.models.TransitionMemory(20)

    logits_clip_low = -2
    logits_clip_high = 2

    trainer = PPOTrainer(
        network=ac_network,
        memory=memory,
        gamma=0.95,
        batch_size=5,
        lr_critic=1e-4,
        lr_actor=1e-4,
        optimiser="adam",
        train_every="step",
        update_interval=20,
        clip_eps=0.2,
        c1=1,
        c2=0.01,
        logits_clip_low=logits_clip_low,
        logits_clip_high=logits_clip_high,
    )

    algo = marl.policy_gradient.PPO(
        ac_network=ac_network,
        train_policy=marl.policy.CategoricalPolicy(),
        test_policy=marl.policy.ArgMax(),
        # extra_policy=marl.policy.ExtraPolicy(env.n_agents),
        # extra_policy_every=50,
        logits_clip_low=logits_clip_low,
        logits_clip_high=logits_clip_high,
    )
    # logdir = f"logs/{env.name}-PPO-5"
    logdir = f"logs/{env.name}-PPO-gamma{trainer.gamma}-steps{n_steps}-EP_{algo.extra_policy != None}-clip4"
    if args.debug:
        logdir = "logs/debug"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_lle(args: Arguments):
    n_steps = 300_000
    test_interval = 5000
    gamma = 0.95
    from marl.env.zero_punishment import ZeroPunishment
    from marl.env.random_initial_pos import RandomInitialPos
    from marl.env.b_shaping import BShaping

    # file = "maps/1b"
    # file = "maps/lvl6-no-gems"
    builder = LLE.from_file(args.map_file)
    lle = builder.obs_type(ObservationType.LAYERED).build()
    env = lle
    env = RandomInitialPos(env, 0, 1, 0, lle.width - 1)
    env = BShaping(env, lle.world, 1, args.reward_delay, args.reward_in_laser)
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

    if args.logdir is not None:
        if not args.logdir.startswith("logs/"):
            args.logdir = "logs/" + args.logdir
    elif args.debug:
        args.logdir = "logs/debug"
    else:
        args.logdir = f"logs/bottleneck-{args.map_file.replace('maps/', 'map=')}-delay={args.reward_delay}"
        # if trainer.mixer is not None:
        #     args.logdir += f"-{trainer.mixer.name}"
        # else:
        #     args.logdir += "-iql"
        # if trainer.ir_module is not None:
        #     args.logdir += f"-{trainer.ir_module.name}"
        # if isinstance(trainer.memory, marl.models.PrioritizedMemory):
        #     args.logdir += "-PER"
    return marl.Experiment.create(
        args.logdir,
        algo=algo,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
    )


def create_lle_baseline(args: Arguments):
    # use Episode update -> use reshape in the nn
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    obs_type = lle.ObservationType.LAYERED
    env = lle.LLE.level(6, obs_type=obs_type, state_type=lle.ObservationType.FLATTENED, multi_objective=False)
    env = rlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    test_env = None
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.EpisodeMemory(5000)
    steps_eps = 200_000
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=steps_eps,
    )
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=32,
        train_interval=(1, "episode"),
        gamma=gamma,
        mixer=marl.qlearning.VDN(env.n_agents),
        grad_norm_clipping=10,
        ir_module=None,
    )

    algo = marl.qlearning.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.name is not None:
        logdir = f"logs/{args.name}"
    elif args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/baseline-qnetwork-eps{steps_eps}-{env.name}-{obs_type}"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}"
        else:
            logdir += "-iql"
        if trainer.ir_module is not None:
            logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(
        logdir,
        algo=algo,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
        test_env=test_env,
    )


def create_lle_maic(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    obs_type = lle.ObservationType.PARTIAL_7x7
    env = lle.LLE.level(6, obs_type, state_type=lle.ObservationType.FLATTENED, multi_objective=False)
    env = rlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    # TODO : improve args
    opt = SimpleNamespace()
    opt.n_agents = env.n_agents
    opt.latent_dim = 8
    opt.nn_hidden_size = 64
    opt.rnn_hidden_dim = 64
    opt.attention_dim = 32
    opt.var_floor = 0.002
    opt.mi_loss_weight = 0.001
    opt.entropy_loss_weight = 0.01

    gamma = 0.95
    eps_steps = 200_000
    # Add the MAICNetwork (MAICAgent)
    maic_network = marl.nn.model_bank.MAICNetwork.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        eps_steps,
    )
    # Add the MAICAlgo (MAICMAC)
    algo = marl.qlearning.MAICAlgo(maic_network=maic_network, train_policy=train_policy, test_policy=marl.policy.ArgMax(), args=opt)
    batch_size = 32
    # Add the MAICTrainer (MAICLearner)
    trainer = MAICTrainer(
        args=opt,
        maic_network=maic_network,
        train_policy=train_policy,
        batch_size=batch_size,
        memory=memory,
        gamma=gamma,
        mixer=marl.qlearning.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents), #TODO: try with QMix : state needed
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        grad_norm_clipping=10,
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/MAIC-{batch_size}-eps{eps_steps}-{env.name}-{obs_type}"
        if trainer.double_qlearning:
            logdir += "-double"
        else:
            logdir += "-single"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}"
        else:
            logdir += "-iql"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=test_interval, n_steps=n_steps)


def create_lle_maicRQN(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    obs_type = lle.ObservationType.PARTIAL_7x7
    env = lle.LLE.level(6, obs_type, state_type=lle.ObservationType.FLATTENED, multi_objective=False)
    env = rlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    # TODO : improve args
    opt = SimpleNamespace()
    opt.n_agents = env.n_agents
    opt.latent_dim = 8
    opt.nn_hidden_size = 64
    opt.rnn_hidden_dim = 64
    opt.attention_dim = 32
    opt.var_floor = 0.002
    opt.mi_loss_weight = 0.001
    opt.entropy_loss_weight = 0.01

    gamma = 0.95
    # Add the MAICNetwork (MAICAgent)
    qnetwork = marl.nn.model_bank.MAICNetworkRDQN.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    eps_steps = 200_000
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, eps_steps)
    bs = 32
    trainer = DQNTrainer(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=bs,
        train_interval=(1, "episode"),
        gamma=gamma,
        mixer=marl.qlearning.VDN(env.n_agents),
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
        logdir = f"logs/MAICRQN-NoComm--{bs}-eps{eps_steps}-{env.name}-{obs_type}"
        if trainer.double_qlearning:
            logdir += "-double"
        else:
            logdir += "-single"
        if trainer.mixer is not None:
            logdir += f"-{trainer.mixer.name}"
        else:
            logdir += "-iql"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=test_interval, n_steps=n_steps)


def main(args: Arguments):
    try:
        # exp = create_smac(args)
        # exp = create_ppo_lle()
        # exp = create_lle(args)
        exp = create_lle_baseline(args)
        # exp = create_lle_maic(args)
        # exp = create_lle_maicRQN(args)
        print(exp.logdir)
        shutil.copyfile(__file__, exp.logdir + "/create_experiment.py")
        if args.run:
            args.logdir = exp.logdir
            run_experiment(args)
            # exp.create_runner(seed=0).to("auto").train(args.n_tests)
    except ExperimentAlreadyExistsException as e:
        if not args.overwrite:
            response = ""
            response = input(f"Experiment already exists in {e.logdir}. Overwrite? [y/n] ")
            if response.lower() != "y":
                print("Experiment not created.")
                return
        shutil.rmtree(e.logdir)
        return main(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
