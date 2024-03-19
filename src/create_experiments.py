import shutil
import marl
import lle
import rlenv
from typing import Optional
import typed_argparse as tap
from lle import WorldState
from marl.training import DQNTrainer
from marl.training.ppo_trainer import PPOTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.utils import ExperimentAlreadyExistsException

from run import Arguments as RunArguments, main as run_experiment


class Arguments(tap.TypedArgs):
    name: Optional[str] = tap.arg(default=None, help="Name of the experimentto create (overrides 'debug').")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")
    n_tests: int = tap.arg(default=1, help="Number of tests to run")
    n_runs: int = tap.arg(default=1, help="Number of runs to start. Only applies if 'run' is True.")


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
    env = lle.LLE.level(6, lle.ObservationType.LAYERED, state_type=lle.ObservationType.FLATTENED, multi_objective=True)
    env = curriculum(env, n_steps)
    env = rlenv.Builder(env).agent_id().time_limit(78, add_extra=False).build()
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
    return marl.Experiment.create(logdir, algo=algo, trainer=trainer, env=env, test_interval=test_interval, n_steps=n_steps)


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
