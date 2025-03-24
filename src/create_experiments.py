import shutil
import marl
import marlenv
from marlenv import MARLEnv, DiscreteActionSpace
from typing import Any, Optional, Literal
import typed_argparse as tap
from marl.training import DQNTrainer
from marl.training.qtarget_updater import SoftUpdate, HardUpdate
from marl.exceptions import ExperimentAlreadyExistsException

from lle import LLE
from run import Arguments as RunArguments, main as run_experiment
from marl.nn.mixers import VDN
from marl.training.intrinsic_reward import AdvantageIntrinsicReward
from marl.training.ppo_trainer import PPOTrainer
from marl.training.haven_trainer import HavenTrainer
from marl.nn.model_bank.actor_critics import CNNContinuousActorCritic


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
        mixer=marl.nn.mixers.QPlex(
            n_agents=env.n_agents,
            n_actions=env.n_actions,
            state_size=env.state_shape[0],
            adv_hypernet_embed=64,
            n_heads=10,
            weighted_head=True,
        ),
        grad_norm_clipping=10,
    )

    algo = marl.agents.RDQN(
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
    return marl.Experiment.create(logdir=logdir, agent=algo, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


def make_haven(agent_type: Literal["dqn", "ppo"], ir: bool):
    n_steps = 2_000_000
    test_interval = 5000
    WARMUP_STEPS = 200_000
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
            meta_agent = PPOTrainer(
                actor_critic=CNNContinuousActorCritic(
                    input_shape=meta_env.observation_shape,
                    n_extras=meta_env.extras_shape[0],
                    action_output_shape=(N_SUBGOALS,),
                ),
                train_interval=1024,
                minibatch_size=64,
                n_epochs=32,
                value_mixer=VDN.from_env(meta_env),
                gamma=gamma,
                lr_actor=5e-4,
                lr_critic=1e-3,
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
        logdir="logs/haven",
    )
    # exp.run()


def make_dqn(
    args: Arguments,
    env: MARLEnv[Any, DiscreteActionSpace],
    test_env: Optional[MARLEnv[Any, DiscreteActionSpace]] = None,
    mixing: Literal["vdn", "qmix", "qplex"] = "vdn",
    gamma=0.95,
    n_steps=1_000_000,
    test_interval=5000,
):
    match mixing:
        case "vdn":
            mixer = VDN.from_env(env)
        case "qmix":
            mixer = marl.nn.mixers.QMix.from_env(env)
        case "qplex":
            mixer = marl.nn.mixers.QPlex.from_env(env)
        case other:
            raise ValueError(f"Invalid mixer: {other}")
    trainer = DQNTrainer(
        qnetwork=marl.nn.model_bank.IndependentCNN.from_env(env),
        train_policy=marl.policy.EpsilonGreedy.linear(
            1.0,
            0.05,
            n_steps=200_000,
        ),
        memory=marl.models.TransitionMemory(50_000),
        optimiser="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=64,
        train_interval=(5, "step"),
        gamma=gamma,
        mixer=mixer,
        grad_norm_clipping=10,
        ir_module=None,
    )
    algo = trainer.make_agent(marl.policy.ArgMax())

    if args.logdir is not None:
        if not args.logdir.startswith("logs/"):
            args.logdir = "logs/" + args.logdir
    elif args.debug:
        args.logdir = "logs/debug"
    else:
        args.logdir = f"logs/{env.name}-DQN"
        if trainer.mixer is not None:
            args.logdir += f"-{trainer.mixer.name}"
        else:
            args.logdir += "-iql"
        if trainer.ir_module is not None:
            args.logdir += f"-{trainer.ir_module.name}"
        if isinstance(trainer.memory, marl.models.PrioritizedMemory):
            args.logdir += "-PER"
    return marl.Experiment.create(
        logdir=args.logdir,
        agent=algo,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
        test_env=test_env,
    )


def make_ppo(args: Arguments, env: MARLEnv[Any, DiscreteActionSpace], test_env=None, n_steps=1_000_000, test_interval=5000):
    actor_critic = marl.nn.model_bank.actor_critics.CNN_ActorCritic.from_env(env)
    trainer = PPOTrainer(
        actor_critic=actor_critic,
        train_interval=4096,
        minibatch_size=256,
        n_epochs=64,
        value_mixer=VDN.from_env(env),
        gamma=0.95,
        lr_actor=5e-4,
        lr_critic=1e-3,
        exploration_c2=0.005,
    )
    if args.logdir is not None:
        if not args.logdir.startswith("logs/"):
            args.logdir = "logs/" + args.logdir
    elif args.debug:
        args.logdir = "logs/debug"
    else:
        args.logdir = f"logs/{env.name}-PPO"
        if trainer.value_mixer is not None:
            args.logdir += f"-{trainer.value_mixer.name}"
        else:
            args.logdir += "-iql"
    return marl.Experiment.create(
        logdir=args.logdir,
        trainer=trainer,
        env=env,
        test_interval=test_interval,
        n_steps=n_steps,
        test_env=test_env,
    )


def make_lle(args: Arguments):
    # lle = RandomizedLasers(
    env = (
        LLE.level(6)
        # LLE.from_file("maps/lvl6-start-above.toml")
        .obs_type("layered")
        .state_type("state")
        # .pbrs(
        #    gamma=gamma,
        #    reward_value=1,
        #    lasers_to_reward=[(4, 0), (6, 12)],
        # )
        .builder()
        .agent_id()
        .time_limit(78, add_extra=True)
        .build()
    )
    test_env = None
    return env, test_env


def create_overcooked(args: Arguments):
    horizon = 400
    env = marlenv.adapters.Overcooked.from_layout("bottleneck", horizon)
    env = marlenv.Builder(env).agent_id().build()
    return make_ppo(args, env)


def main(args: Arguments):
    try:
        # exp = create_smac(args)
        env, test_env = make_lle(args)
        # exp = make_dqn(args, env, test_env)
        exp = make_ppo(args, env, test_env, n_steps=4_000_000)
        # exp = create_overcooked(args)
        # exp = make_haven("dqn", ir=True)
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
