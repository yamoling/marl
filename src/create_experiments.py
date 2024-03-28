import shutil
import marl
import lle
import rlenv
from typing import Optional
import typed_argparse as tap
from marl.training import DQNTrainer, DDPGTrainer, PPOTrainer, CNetTrainer, MAICTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.utils import ExperimentAlreadyExistsException
from marl.utils import MultiSchedule, LinearSchedule
from lle import WorldState
from run import Arguments as RunArguments, main as run_experiment
from types import SimpleNamespace
from copy import deepcopy


class Arguments(tap.TypedArgs):
    name: Optional[str] = tap.arg(default=None, help="Name of the experimentto create (overrides 'debug').")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    n_runs: int = tap.arg(default=1, help="Number of runs to start. Only applies if 'run' is True.")


class Args:
    def __init__(self):
        self.latent_dim = 64
        self.hidden_size = 128
        self.rnn_hidden_dim = 128
        self.attention_dim = 64
        self.var_floor = 1e-6


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
    n_steps = 10_000
    env = lle.LLE.level(2, lle.ObservationType.LAYERED)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()

    ac_network = marl.nn.model_bank.CNN_DActor_CCritic.from_env(env)
    memory = marl.models.TransitionMemory(50_000)

    trainer = DDPGTrainer(
        network=ac_network, memory=memory, batch_size=4, optimiser="adam", train_every="step", update_interval=20, gamma=0.99
    )

    algo = marl.policy_gradient.DDPG(ac_network=ac_network)
    logdir = f"logs/{env.name}-TEST-DDPG"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_ppo_lle(args: Arguments):
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


def curriculum(env: lle.LLE, n_steps: int):
    world = env.world
    env.reset()
    i_positions = list(range(world.height - 1, -1, -1))
    j_positions = [3] * len(i_positions)
    initial_states = list[WorldState]()
    exclude_i = [6]
    for i, j in zip(i_positions, j_positions):
        if i in exclude_i:
            continue
        start_positions = [(i, j + n) for n in range(world.n_agents)]
        initial_states.append(WorldState(start_positions, [False] * world.n_gems))
    interval = n_steps // len(initial_states)
    return marl.env.CurriculumLearning(env, initial_states, interval)


def create_lle(args: Arguments):
    n_steps = 1_000_000
    test_interval = 5000
    gamma = 0.95
    # envs = [lle.LLE.level(i, lle.ObservationType.LAYERED_PADDED, state_type=lle.ObservationType.FLATTENED) for i in range(1, 7)]
    # env = marl.env.EnvPool(envs)
    env = lle.LLE.level(6, lle.ObservationType.PARTIAL_7x7, state_type=lle.ObservationType.FLATTENED, multi_objective=False)
    # width, height = env.width, env.height
    # env = curriculum(env, n_steps)
    # env = marl.env.lle_curriculum.RandomInitialStates(env, True)
    # from marl.env import ExtraObjective

    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=True).build()
    # test_env = lle.LLE.level(6, lle.ObservationType.LAYERED, state_type=lle.ObservationType.FLATTENED, multi_objective=False)
    # test_env = rlenv.Builder(test_env).agent_id().time_limit(78, add_extra=True).build()
    test_env = None
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    # eps_schedule = MultiSchedule(
    #     {
    #         0: LinearSchedule(1, 0.05, 150_000),
    #         300_000: LinearSchedule(1, 0.05, 150_000),
    #         600_000: LinearSchedule(1, 0.05, 150_000),
    #     }
    # )
    # train_policy = marl.policy.EpsilonGreedy(eps_schedule)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        n_steps=200_000,
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
        logdir = f"logs/curriculum-{env.name}"
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
    n_steps = 600_000
    env = lle.LLE.level(2, lle.ObservationType.FLATTENED)
    env = rlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2, add_extra=False).build()
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

    # Add the MAICNetwork (MAICAgent)
    maic_network = marl.nn.model_bank.MAICNetwork.from_env(env, opt)
    memory = marl.models.EpisodeMemory(5000)
    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        300_000,
    )
    # Add the MAICAlgo (MAICMAC)
    algo = marl.qlearning.MAICAlgo(maic_network=maic_network, train_policy=train_policy, test_policy=marl.policy.ArgMax(), args=opt)
    batch_size = 64
    # Add the MAICTrainer (MAICLearner)
    trainer = MAICTrainer(
        args=opt,
        maic_algo=algo,
        train_policy=train_policy,
        batch_size=batch_size,
        memory=memory,
        mixer=marl.qlearning.VDN(env.n_agents),
        # mixer=marl.qlearning.QMix(env.n_agents), TODO: try with QMix : state needed
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        grad_norm_clipping=10,
    )

    if args.debug:
        logdir = "logs/debug"
    else:
        logdir = f"logs/MAIC{batch_size}-{env.name}"
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
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


## Init RIAL/DIAL options https://github.com/minqi/learning-to-communicate-pytorch
def init_action_and_comm_bits(opt):
    opt.comm_enabled = opt.game_comm_bits > 0 and opt.game_nagents > 1
    if opt.model_comm_narrow is None:
        opt.model_comm_narrow = opt.model_dial
    if not opt.model_comm_narrow and opt.game_comm_bits > 0:
        opt.game_comm_bits = 2**opt.game_comm_bits
    if opt.comm_enabled:
        opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
    else:
        opt.game_action_space_total = opt.game_action_space
    return opt


def init_opt(opt):
    if not opt.model_rnn_layers:
        opt.model_rnn_layers = 2
    if opt.model_avg_q is None:
        opt.model_avg_q = True
    if opt.eps_decay is None:
        opt.eps_decay = 1.0
    opt = init_action_and_comm_bits(opt)
    return opt


def create_lle_rial_dial(args: Arguments):
    n_steps = 300_000
    env = lle.LLE.level(2, lle.ObservationType.FLATTENED)
    nsteps_limit = env.width * env.height // 2
    env = rlenv.Builder(env).agent_id().time_limit(nsteps_limit, add_extra=False).build()

    opt = SimpleNamespace()
    # TODO : Add options from Json ?
    opt.game = "LLE"
    opt.game_nagents = env.n_agents
    opt.game_action_space = env.n_actions
    opt.game_comm_limited = False
    opt.game_comm_bits = 3  # test with different values
    opt.game_comm_sigma = 2
    opt.nsteps = nsteps_limit
    opt.gamma = 0.9
    opt.model_dial = True
    opt.model_target = True
    opt.model_bn = True
    opt.model_know_share = True
    opt.model_action_aware = True
    opt.model_rnn_size = 128
    opt.bs = 32
    opt.learningrate = 0.0005
    opt.momentum = 0.05
    opt.eps = 0.05
    opt.nepisodes = n_steps
    opt.step_test = 10
    opt.step_target = 100
    opt.cuda = 0

    opt.model_rnn_layers = 2
    opt.model_avg_q = True
    opt.eps_decay = 1.0
    opt.model_comm_narrow = opt.model_dial

    opt.model_rnn_dropout_rate = 0.0
    opt.game_comm_hard = False

    opt = init_opt(opt)

    cnet = marl.nn.model_bank.CNet.from_env(env, opt)

    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        100_000,
    )

    algo = marl.qlearning.CNetAlgo(opt, cnet, deepcopy(cnet), train_policy, marl.policy.ArgMax())

    trainer = CNetTrainer(opt, algo)

    if args.debug:
        logdir = "logs/debug"
    else:
        name = "DIAL" if opt.model_dial else "RIAL"
        logdir = f"logs/{name}{opt.bs}-{env.name}"
        # if trainer.mixer is not None:
        #     logdir += f"-{trainer.mixer.name}"
        # else:
        logdir += "-iql"
        # if isinstance(trainer.memory, marl.models.PrioritizedMemory):
        #     logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def create_lle_rial_dial_bs_1(args: Arguments):
    n_steps = 300_000
    env = lle.LLE.level(2, lle.ObservationType.FLATTENED)
    nsteps_limit = env.width * env.height // 2
    env = rlenv.Builder(env).agent_id().time_limit(nsteps_limit, add_extra=False).build()

    opt = SimpleNamespace()
    # TODO : Add options from Json ?
    opt.game = "LLE"
    opt.game_nagents = env.n_agents
    opt.game_action_space = env.n_actions
    opt.game_comm_limited = False
    opt.game_comm_bits = 3  # test with different values
    opt.game_comm_sigma = 2
    opt.nsteps = nsteps_limit
    opt.gamma = 0.9
    opt.model_dial = True
    opt.model_target = True
    opt.model_bn = True
    opt.model_know_share = True
    opt.model_action_aware = True
    opt.model_rnn_size = 128
    opt.bs = 1
    opt.learningrate = 0.0005
    opt.momentum = 0.05
    opt.eps = 0.05
    opt.nepisodes = n_steps
    opt.step_test = 10
    opt.step_target = 100
    opt.cuda = 0

    opt.model_rnn_layers = 2
    opt.model_avg_q = True
    opt.eps_decay = 1.0
    opt.model_comm_narrow = opt.model_dial

    opt.model_rnn_dropout_rate = 0.0
    opt.game_comm_hard = False

    opt = init_opt(opt)

    cnet = marl.nn.model_bank.CNet.from_env(env, opt)

    train_policy = marl.policy.EpsilonGreedy.linear(
        1.0,
        0.05,
        100_000,
    )

    algo = marl.qlearning.CNetAlgo(opt, cnet, deepcopy(cnet), train_policy, marl.policy.ArgMax())

    trainer = CNetTrainer(opt, algo)

    if args.debug:
        logdir = "logs/debug"
    else:
        name = "DIAL" if opt.model_dial else "RIAL"
        logdir = f"logs/{name}{opt.bs}-{env.name}"
        # if trainer.mixer is not None:
        #     logdir += f"-{trainer.mixer.name}"
        # else:
        logdir += "-iql"
        # if isinstance(trainer.memory, marl.models.PrioritizedMemory):
        #     logdir += "-PER"
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def main(args: Arguments):
    try:
        # exp = create_smac(args)
        # exp = create_ppo_lle()
        exp = create_lle(args)
        print(exp.logdir)
        if args.run:
            run_args = RunArguments(
                logdir=exp.logdir,
                n_tests=args.n_tests,
                seed=0,
                n_runs=args.n_runs,
            )
            run_experiment(run_args)
            # exp.create_runner(seed=0).to("auto").train(args.n_tests)
    except ExperimentAlreadyExistsException as e:
        response = ""
        response = input(f"Experiment already exists in {e.logdir}. Overwrite? [y/n] ")
        if response.lower() != "y":
            print("Experiment not created.")
            return
        shutil.rmtree(e.logdir)
        return main(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
