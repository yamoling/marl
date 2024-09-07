import shutil
import marl
import marlenv
from typing import Optional
import typed_argparse as tap
from marl.training import DQNTrainer, DDPGTrainer, PPOTrainer, MAICTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.exceptions import ExperimentAlreadyExistsException
from marl.algo.qlearning.maic import MAICParameters
from lle import LLE, ObservationType
from run import Arguments as RunArguments, main as run_experiment
from marl.utils import Schedule


class Arguments(RunArguments):
    logdir: Optional[str] = tap.arg(default=None, help="The experiment directory")
    overwrite: bool = tap.arg(default=False, help="Override the existing experiment directory")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")


def create_smac(args: Arguments):
    n_steps = 2_000_000
    env = marlenv.adapters.SMAC("3s_vs_5z")
    env = marlenv.Builder(env).agent_id().last_action().build()
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
        mixer=marl.algo.QPlex(
            n_agents=env.n_agents,
            n_actions=env.n_actions,
            state_size=env.state_shape[0],
            adv_hypernet_embed=64,
            n_heads=10,
            weighted_head=True,
        ),
        grad_norm_clipping=10,
    )

    algo = marl.algo.RDQN(
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
    env = LLE.level(2).obs_type(ObservationType.LAYERED).state_type(ObservationType.LAYERED).build()
    env = marlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.DDPG_NN_TEST.from_env(env)
    memory = marl.models.TransitionMemory(50_000)

    train_policy = marl.policy.NoisyCategoricalPolicy()
    test_policy = marl.policy.ArgMax()

    trainer = DDPGTrainer(
        network=ac_network, memory=memory, batch_size=64, optimiser="adam", train_every="step", update_interval=5, gamma=0.95, lr=1e-5
    )

    algo = marl.algo.DDPG(ac_network=ac_network, train_policy=train_policy, test_policy=test_policy)
    # logdir = f"logs/{env.name}-TEST-DDPG"
    logdir = "logs/ddpg_lvl2_lr_1e-5"
    if args.debug:
        logdir = "logs/debug"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_ppo_lle(args: Arguments):
    n_steps = 300_000
    walkable_lasers = True
    temperature = 10
    # env = LLE.from_file("maps/lvl3_without_gem").obs_type(ObservationType.LAYERED).walkable_lasers(walkable_lasers).build()
    env = LLE.level(3).obs_type(ObservationType.LAYERED).walkable_lasers(walkable_lasers).build()
    env = marlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
    ac_network.temperature = temperature

    entropy_schedule = Schedule.linear(0.05, 0.001, round(2 / 3 * n_steps))
    temperature_schedule = Schedule.linear(50, 1, round(2 / 3 * n_steps))

    # ac_network = marl.nn.model_bank.Clipped_CNN_ActorCritic.from_env(env)
    memory = marl.models.TransitionMemory(20)

    logits_clip_low = -2.0
    logits_clip_high = 2.0

    trainer = PPOTrainer(
        network=ac_network,
        memory=memory,
        gamma=0.99,
        batch_size=2,
        lr_critic=1e-4,
        lr_actor=1e-4,
        optimiser="adam",
        train_every="step",
        update_interval=8,
        n_epochs=4,
        clip_eps=0.2,
        c1=0.5,
        c2=0.01,
        c2_schedule=entropy_schedule,
        softmax_temp_schedule=temperature_schedule,
        logits_clip_low=logits_clip_low,
        logits_clip_high=logits_clip_high,
    )

    algo = marl.algo.PPO(
        ac_network=ac_network,
        train_policy=marl.policy.CategoricalPolicy(),
        #     train_policy=marl.policy.EpsilonGreedy.linear(
        #     1.0,
        #     0.05,
        #     n_steps=300_000,
        # ),
        test_policy=marl.policy.ArgMax(),
        # extra_policy=marl.policy.ExtraPolicy(env.n_agents),
        # extra_policy_every=50,
        logits_clip_low=logits_clip_low,
        logits_clip_high=logits_clip_high,
    )

    # logdir = f"logs/PPO-{env.name}-batch_{trainer.update_interval}_{trainer.batch_size}-gamma_{trainer.gamma}-WL_{walkable_lasers}-C2_{trainer.c2}-C1_{trainer.c1}"
    # logdir += "-epsGreedy" if isinstance(algo.train_policy, marl.policy.EpsilonGreedy) else ""
    # logdir += "-clipped" if isinstance(ac_network, marl.nn.model_bank.Clipped_CNN_ActorCritic) else ""
    logdir = "logs/ppo_lvl3_default"
    if args.debug:
        logdir = "logs/debug"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_lle(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    env = LLE.level(6).obs_type(ObservationType.LAYERED).state_type(ObservationType.STATE).build()
    env = marlenv.Builder(env).centralised().time_limit(78, add_extra=True).build()
    test_env = None

    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=100_000,
    )
    mixer = marl.algo.VDN.from_env(env)
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
        mixer=mixer,
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents),
        grad_norm_clipping=10,
        # ir_module=rnd,
    )

    algo = marl.algo.DQN(
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
        args.logdir = f"logs/{env.name}"
        if trainer.mixer is not None:
            args.logdir += f"-{trainer.mixer.name}"
        else:
            args.logdir += "-iql"
        if trainer.ir_module is not None:
            args.logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            args.logdir += "-PER"
    return marl.Experiment.create(
        args.logdir,
        algo=algo,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
        test_env=test_env,
    )


def create_lle_baseline(args: Arguments):
    # use Episode update -> use reshape in the nn
    n_steps = 500_000
    test_interval = 5000
    gamma = 0.95
    obs_type = ObservationType.LAYERED
    env = LLE.level(3).obs_type(obs_type).state_type(ObservationType.FLATTENED).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    test_env = None
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.EpisodeMemory(5000)
    steps_eps = 500_000
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
        batch_size=64,
        train_interval=(1, "episode"),
        gamma=gamma,
        mixer=marl.algo.VDN(env.n_agents),
        grad_norm_clipping=10,
        ir_module=None,
    )

    algo = marl.algo.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.logdir is not None:
        logdir = f"logs/{args.logdir}"
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
    n_steps = 200_000
    test_interval = 5000
    obs_type = ObservationType.PARTIAL_7x7
    env = LLE.level(2).obs_type(obs_type).state_type(ObservationType.FLATTENED).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    # TODO : improve args
    opt = MAICParameters(n_agents=env.n_agents, com=True)

    gamma = 0.95
    eps_steps = 50_000
    # Add the MAICNetwork (MAICAgent)
    maic_network = marl.nn.model_bank.MAICNetwork.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        eps_steps,
    )
    # Add the MAICAlgo (MAICMAC)
    algo = marl.algo.MAIC(maic_network=maic_network, train_policy=train_policy, test_policy=marl.policy.ArgMax(), args=opt)
    batch_size = 32
    # Add the MAICTrainer (MAICLearner)
    trainer = MAICTrainer(
        args=opt,
        maic_network=maic_network,
        train_policy=train_policy,
        batch_size=batch_size,
        memory=memory,
        gamma=gamma,
        mixer=marl.algo.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.state_shape[0], env.n_agents), #TODO: try with QMix : state needed
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        grad_norm_clipping=10,
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        name = "MAIC-NoCOM" if not opt.com else "MAIC"
        logdir = f"logs/{name}-{batch_size}-eps{eps_steps}-{env.name}-{obs_type}"
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


def create_lle_maicRDQN(args: Arguments):
    n_steps = 2_000_000
    test_interval = 5000
    obs_type = ObservationType.PARTIAL_7x7
    env = LLE.level(6).obs_type(obs_type).state_type(ObservationType.FLATTENED).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    # TODO : improve args
    opt = MAICParameters(n_agents=env.n_agents, com=True)

    gamma = 0.95
    qnetwork = marl.nn.model_bank.MAICNetworkRDQN.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    eps_steps = 200_000
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, eps_steps)
    bs = 64
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
        mixer=marl.algo.VDN(env.n_agents),
        grad_norm_clipping=10,
        ir_module=None,
    )

    algo = marl.algo.RDQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        name = "MAICRDQN-NoCOM" if not opt.com else "MAICRDQN"
        logdir = f"logs/{name}-{bs}-eps{eps_steps}-steps{n_steps}-{env.name}-{obs_type}"
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


def create_lle_maicCNN(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    obs_type = ObservationType.PARTIAL_7x7
    env = LLE.level(6).obs_type(obs_type).state_type(ObservationType.FLATTENED).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    # TODO : improve args
    opt = MAICParameters(n_agents=env.n_agents, com=True)

    gamma = 0.95
    qnetwork = marl.nn.model_bank.MAICNetworkCNN.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    eps_steps = 200_000
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, eps_steps)
    bs = 64
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
        mixer=marl.algo.VDN(env.n_agents),
        grad_norm_clipping=10,
        ir_module=None,
    )

    algo = marl.algo.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        name = "MAICCNN-NoCOM" if not opt.com else "MAICCNN"
        logdir = f"logs/{name}-{bs}-eps{eps_steps}-{env.name}-{obs_type}"
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


def create_lle_maicCNNRDQN(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    obs_type = ObservationType.PARTIAL_7x7
    env = LLE.level(4).obs_type(obs_type).state_type(ObservationType.FLATTENED).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=True).build()
    # TODO : improve args
    opt = MAICParameters(n_agents=env.n_agents, com=True)

    gamma = 0.95
    qnetwork = marl.nn.model_bank.MAICNetworkCNNRDQN.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    eps_steps = 150_000
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, eps_steps)
    bs = 64
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
        mixer=marl.algo.VDN(env.n_agents),
        grad_norm_clipping=10,
        ir_module=None,
    )

    algo = marl.algo.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        test_policy=marl.policy.ArgMax(),
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        name = "MAICCNNRDQN-NoCOM" if not opt.com else "MAICCNNDRQN"
        logdir = f"logs/{name}-{bs}-eps{eps_steps}-{env.name}-{obs_type}"
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
        # exp = create_ddpg_lle(args)
        exp = create_lle(args)
        # exp = create_lle(args)
        # exp = create_lle_maic(args)
        # exp = create_lle_maicRQN(args)
        print(exp.logdir)
        shutil.copyfile(__file__, exp.logdir + "/tmp.py")
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
