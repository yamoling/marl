import shutil
from copy import deepcopy
from typing import Literal, Optional

import marlenv
import typed_argparse as tap
from lle import LLE
from marlenv import MARLEnv, MultiDiscreteSpace
from marlenv.utils import Schedule

import marl
from marl import Trainer
from marl.env import StateCounter
from marl.exceptions import ExperimentAlreadyExistsException
from marl.nn.mixers import VDN
from marl.nn.model_bank.actor_critics import CNNContinuousActorCritic
from marl.optimism import VBE
from marl.training import DQN, PPO, SoftUpdate
from marl.training.haven_trainer import HavenTrainer
from marl.training.intrinsic_reward import AdvantageIntrinsicReward
from run import Arguments as RunArguments
from run import main as run_experiment


class Arguments(RunArguments):
    logdir: Optional[str] = tap.arg(default=None, help="The experiment directory")
    overwrite: bool = tap.arg(default=False, help="Override the existing experiment directory")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")


def make_smac(map_name: str):
    env = marlenv.adapters.SMAC(map_name)
    env = marlenv.Builder(env).agent_id().build()
    return env, None


def create_smac(args: Arguments):
    n_steps = 2_000_000
    # env = marlenv.adapters.SMAC("3s_vs_5z")
    env = marlenv.adapters.SMAC("3m")
    env = marlenv.Builder(env).agent_id().last_action().build()
    qnetwork = marl.nn.model_bank.RNNQMix.from_env(env)
    memory = marl.models.TransitionMemory(50_000)
    train_policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000)
    trainer = DQN(
        qnetwork,
        train_policy=train_policy,
        memory=memory,
        double_qlearning=True,
        target_updater=SoftUpdate(),
        lr=5e-4,
        optimiser="adam",
        batch_size=32,
        train_interval=(1, "step"),
        gamma=0.99,
        mixer=VDN.from_env(env),
        grad_norm_clipping=10,
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
    return marl.Experiment.create(logdir=logdir, trainer=trainer, env=env, test_interval=5000, n_steps=n_steps)


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
            meta_agent = PPO(
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
            meta_agent = DQN(
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
                ir_module=ir_module,
                gamma=gamma,
                mixer=VDN.from_env(meta_env),
                grad_norm_clipping=10.0,
            )
        case other:
            raise ValueError(f"Invalid agent type: {other}")

    env = marlenv.Builder(meta_env).pad("extra", N_SUBGOALS).build()
    worker_trainer = DQN(
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


def make_mixer(env: MARLEnv[MultiDiscreteSpace], mixing: Optional[Literal["vdn", "qmix", "qplex"]] = "vdn"):
    match mixing:
        case None:
            mixer = None
        case "vdn":
            mixer = VDN.from_env(env)
        case "qmix":
            mixer = marl.nn.mixers.QMix.from_env(env)
        case "qplex":
            mixer = marl.nn.mixers.QPlex.from_env(env)
        case other:
            raise ValueError(f"Invalid mixer: {other}")
    return mixer


def make_dqn(
    env: MARLEnv[MultiDiscreteSpace],
    mixing: Optional[Literal["vdn", "qmix", "qplex"]] = "vdn",
    ir_method: Optional[Literal["rnd", "tomir", "icm"]] = None,
    gamma=0.95,
    noisy: bool = False,
    use_vbe: bool = False,
):
    mixer = make_mixer(env, mixing)
    qnetwork = marl.nn.model_bank.MLP.from_env(env, hidden_sizes=(128, 128))
    match ir_method:
        case "rnd":
            ir = marl.training.intrinsic_reward.RandomNetworkDistillation.from_env(env)
        case "tomir":
            ir = marl.training.intrinsic_reward.ToMIR.from_env(env, qnetwork, ir_weight=0.05, is_individual=mixer is None)
        case "icm":
            ir = marl.training.intrinsic_reward.ICM.from_env(env)
        case None:
            ir = None
    if noisy:
        policy = marl.policy.ArgMax()
    else:
        policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=50_000)
    vbe = None
    if use_vbe:
        vbe = VBE(gamma, deepcopy(qnetwork), 3, 1e-4)
    return DQN(
        qnetwork=qnetwork,
        train_policy=policy,
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
        ir_module=ir,
        vbe=vbe,
    )


def make_ppo(
    env: MARLEnv[MultiDiscreteSpace],
    mixing: Optional[Literal["vdn", "qmix", "qplex"]] = None,
    eps_clip: float = 0.2,
    train_interval: int = 400,
    k: int = 20,
    minibatch_size: int = 400,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    c1: float = 0.5,
    c2: float = 0.1,
    grad_norm_clipping: float = 0.1,
):
    match env.observation_shape:
        case (_, _, _):
            actor_critic = marl.nn.model_bank.actor_critics.CNN_ActorCritic.from_env(env)
        case (_,):
            actor_critic = marl.nn.model_bank.actor_critics.SimpleActorCritic.from_env(env)
        case other:
            raise ValueError(f"Invalid observation shape: {other}")
    mixer = make_mixer(env, mixing)
    return PPO(
        actor_critic=actor_critic,
        train_interval=train_interval,
        n_epochs=k,
        minibatch_size=minibatch_size,
        value_mixer=mixer,
        gamma=0.99,
        eps_clip=eps_clip,
        gae_lambda=0.98,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        critic_c1=c1,
        exploration_c2=c2,
        grad_norm_clipping=grad_norm_clipping,
    )


def make_experiment(
    args: Arguments,
    trainer: Trainer,
    env: MARLEnv,
    test_env: Optional[MARLEnv],
    n_steps: int,
):
    if args.logdir is not None:
        if not args.logdir.startswith("logs/"):
            args.logdir = "logs/" + args.logdir
    elif args.debug:
        args.logdir = "logs/debug"
    else:
        args.logdir = f"logs/{env.name}-{trainer.name}"
        match trainer:
            case DQN():
                if trainer.mixer is not None:
                    args.logdir += f"-{trainer.mixer.name}"
                else:
                    args.logdir += "-IQL"
                if trainer.ir_module is not None:
                    args.logdir += f"-{trainer.ir_module.name}"
                if isinstance(trainer.memory, marl.models.PrioritizedMemory):
                    args.logdir += "-PER"
            case PPO():
                if trainer.value_mixer is not None:
                    args.logdir += f"-{trainer.value_mixer.name}"
    return marl.Experiment.create(
        logdir=args.logdir,
        trainer=trainer,
        env=env,
        test_interval=5000,
        n_steps=n_steps,
        test_env=test_env,
    )


def make_lle():
    env = LLE.level(6).obs_type("layered").state_type("state").build()
    env = StateCounter(env)
    env = marlenv.Builder(env).agent_id().time_limit(78).build()
    test_env = None
    return env, test_env


def make_deepsea():
    env = marlenv.catalog.DeepSea(25)
    env = marl.env.StateCounter(env)
    env = marl.env.NoReward(env)
    return env, None


def make_overcooked():
    env = marlenv.catalog.Overcooked.from_layout("cramped_room", reward_shaping_factor=Schedule.linear(1.0, 0, 1_000_000))
    env = marlenv.Builder(env).agent_id().build()
    test_env = marlenv.catalog.Overcooked.from_layout("cramped_room", reward_shaping_factor=0)
    test_env = marlenv.Builder(test_env).agent_id().build()
    return env, test_env


def main(args: Arguments):
    try:
        # exp = create_smac(args)
        # env, test_env = make_smac("3m")
        # env, test_env = make_lle()
        env, test_env = make_deepsea()
        # env, test_env = make_overcooked()

        trainer = make_dqn(env, mixing="vdn", ir_method=None, noisy=False, gamma=0.95, use_vbe=True)
        # trainer = make_ppo(env, mixing=None, minibatch_size=128, train_interval=1_000, k=40)
        exp = make_experiment(args, trainer, env, test_env, 200_000)
        print(f"Experiment created in {exp.logdir}")
        # exp = create_overcooked(args)
        # exp = make_haven("dqn", ir=True)
        if args.run:
            args.logdir = exp.logdir
            run_experiment(args)
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
if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
