import logging
import os
import shutil
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from typing import Any, Literal, Optional

import marlenv
import typed_argparse as tap
from lle import LLE
from marlenv import MARLEnv, MultiDiscreteSpace
from marlenv.utils import Schedule

import marl
from marl import ReplayMemory
from marl.exceptions import ExperimentAlreadyExistsException
from marl.nn import model_bank
from marl.nn.mixers import VDN
from marl.nn.model_bank.actor_critics import CNNContinuousActorCritic
from marl.optimism import VBE
from marl.training import DQN, SoftUpdate
from marl.training.haven import HavenTrainer
from marl.training.intrinsic_reward import AdvantageIntrinsicReward
from start_run import Arguments as RunArguments
from start_run import main as run_experiment


class Arguments(tap.TypedArgs):
    overwrite: bool = tap.arg(default=False, help="Override the existing experiment directory")
    run: bool = tap.arg(default=False, help="Run the experiment directly after creating it")
    debug: bool = tap.arg(default=False, help="Create the experiment with name 'debug' (overwritten after each run)")
    _logdir: str | None = tap.arg("--logdir", default=None, help="The experiment directory")
    n_runs: int = tap.arg(default=1, help="Number of runs to create")
    _n_jobs: int | None = tap.arg("--n-jobs", default=None, help="Maximal number of simultaneous processes to use")
    n_tests: int = tap.arg(default=1, help="Number of tests to perform ")
    _device: Literal["auto", "cpu"] | str = tap.arg("--device", default="auto", help="The device to use (auto, cpu or cuda:<gpu_id>)")
    disabled_devices: list[int] = tap.arg(default=[], help="Disabled GPU devices", nargs="*")

    @cached_property
    def logdir(self) -> str:
        if self._logdir is None:
            if self.debug:
                logdir = os.path.join("logs", "debug")
            else:
                logdir = os.path.join("logs", datetime.now().isoformat().replace(":", "-"))
        else:
            logdir = self._logdir
        if not logdir.startswith("logs/"):
            logdir = os.path.join("logs", logdir)
        return logdir

    def as_run_args(self):
        return RunArguments(
            logdir=self.logdir,
            n_runs=self.n_runs,
            _n_jobs=self._n_jobs,
            seed=0,
            n_tests=self.n_tests,
            _device=self._device,
            disabled_devices=self.disabled_devices,
        )


def make_smac(map_name: str):
    from absl import logging

    logging.set_verbosity(logging.WARNING)
    env = marlenv.adapters.SMAC(map_name, debug=False)
    env = marlenv.Builder(env).agent_id().build()
    return env, None


def create_smac_experiment(args: Arguments):
    n_steps = 1_000_000
    # env = marlenv.adapters.SMAC("3s_vs_5z")
    env = marlenv.adapters.SMAC("3m")
    env = marlenv.Builder(env).agent_id().last_action().build()
    trainer = marl.training.PPO(
        marl.nn.model_bank.actor_critics.SimpleRecurrentActorCritic(env.observation_shape[0], env.extras_shape[0], env.n_actions),
        VDN(env.n_agents),
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


def make_option_critic(env: MARLEnv[MultiDiscreteSpace], n_options: int = 8):
    from marl.training.option_critic import OptionCritic

    oc = model_bank.CNNOptionCritic.from_env(env, n_options)
    return OptionCritic(oc, env.n_agents, lr=1e-4, memory_size=50_000)


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
        c, h, w = meta_env.state_shape
        value_network = marl.nn.model_bank.actor_critics.CNNCritic((c, h, w), meta_env.state_extra_shape[0])
        ir_module = AdvantageIntrinsicReward(value_network, gamma)
    else:
        ir_module = None

    match agent_type:
        case "ppo":
            c, h, w = meta_env.observation_shape
            meta_agent = marl.training.PPO(
                actor_critic=CNNContinuousActorCritic(
                    input_shape=(c, h, w),
                    n_extras=meta_env.extras_shape[0],
                    action_output_shape=(N_SUBGOALS,),
                ),
                train_interval=1024,
                minibatch_size=64,
                n_epochs=32,
                mixer=VDN.from_env(meta_env),
                gamma=gamma,
                lr_actor=5e-4,
                lr_critic=1e-3,
                # grad_norm_clipping=10.0,
            )
        case "dqn":
            assert len(meta_env.observation_shape) == 3
            meta_agent = DQN(
                qnetwork=marl.nn.model_bank.qnetworks.QCNN(
                    input_shape=meta_env.observation_shape,
                    extras_size=meta_env.extras_shape[0],
                    output=N_SUBGOALS,
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
        qnetwork=marl.nn.model_bank.qnetworks.QCNN.qnetwork(env),
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
    gamma: float = 0.95,
    noisy: bool = False,
    use_vbe: bool = False,
    memory: Optional[ReplayMemory[Any]] = None,
    update_every: tuple[int, Literal["step", "episode"]] = (5, "step"),
):
    mixer = make_mixer(env, mixing)
    if len(env.observation_shape) == 1:
        qnetwork = marl.nn.model_bank.QMLP.qnetwork(env, hidden_sizes=(128, 128))
    elif len(env.observation_shape) == 3:
        qnetwork = marl.nn.model_bank.IndependentCNN.from_env(env, mlp_sizes=(128, 64))
    else:
        raise NotImplementedError(f"Observation shape {env.observation_shape} not supported")
    match ir_method:
        case "rnd":
            ir = marl.training.intrinsic_reward.RND.from_env(env)
        case "tomir":
            ir = marl.training.intrinsic_reward.ToMIR.from_env(env, qnetwork, ir_weight=0.05, is_individual=mixer is None)
        case "icm":
            ir = marl.training.intrinsic_reward.ICM.from_env(env)
        case None:
            ir = None
    if noisy:
        policy = marl.policy.ArgMax()
    else:
        policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=200_000)
    vbe = None
    if use_vbe:
        vbe = VBE(gamma, deepcopy(qnetwork), 8, 1e-4)
    if memory is None:
        memory = marl.models.TransitionMemory(50_000)
    return DQN(
        qnetwork=qnetwork,
        train_policy=policy,
        memory=memory,
        optimiser_type="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=64,
        train_interval=update_every,
        gamma=gamma,
        mixer=mixer,
        grad_norm_clipping=10,
        ir_module=ir,
        vbe=vbe,
    )


def make_mappo(env: MARLEnv, mixing: Literal["vdn", "qmix", "qplex"] | None = "vdn"):
    match env.observation_shape:
        case (c, h, w):
            ac = marl.nn.model_bank.actor_critics.CNNDiscreteAC((c, h, w), env.extras_shape[0], env.n_actions)
        case (_,):
            ac = marl.nn.model_bank.actor_critics.SimpleRecurrentActorCritic.from_env(env)
        case _:
            raise NotImplementedError()
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
            raise ValueError(f"Invalid mixer type: {other}")
    return marl.training.PPO(ac, mixer, train_on="episode", train_interval=64, minibatch_size=16, n_epochs=5)


def make_lle():
    builder = LLE.level(6).obs_type("layered").state_type("state")
    world = builder._world
    to_reward = [laser for laser in world.laser_sources if laser.agent_id in (0, 1)]
    env = builder.pbrs(lasers_to_reward=to_reward, reward_value=1.0, gamma=1).build()
    env = marlenv.Builder(env).agent_id().time_limit(env.width * env.height // 2).build()
    test_env = None
    return env, test_env


def make_deepsea():
    env = marlenv.catalog.DeepSea(50)
    env = marl.env.StateCounter(env)
    env = marl.env.NoReward(env)
    return env, None


def make_overcooked():
    env = marlenv.catalog.overcooked().from_layout("cramped_room", reward_shaping_factor=Schedule.linear(1.0, 0, 1_000_000))
    env = marlenv.Builder(env).agent_id().build()
    test_env = marlenv.catalog.overcooked().from_layout("cramped_room", reward_shaping_factor=0)
    test_env = marlenv.Builder(test_env).agent_id().build()
    return env, test_env


def main(args: Arguments):
    try:
        # env, test_env = make_lle()

        env = (
            LLE.from_file("maps/four_rooms.toml")
            .obs_type("layered")
            .state_type("state")
            .builder()
            .randomize_actions(1 / 3)
            .agent_id()
            .time_limit(1000)
            .build()
        )

        # env, test_env = make_smac("8m")
        # trainer = make_mappo(env, mixing=None)
        # memory = marl.models.replay_memory.EpisodeMemory(10_000)
        # trainer = make_dqn(env, mixing="qplex", gamma=0.99, memory=memory, update_every=(1, "episode"))
        trainer = make_option_critic(env)
        exp = marl.Experiment.create(
            logdir=args.logdir,
            trainer=trainer,
            env=env,
            test_interval=1000,
            n_steps=1_000_000,
            logger="csv",
        )
        logging.info(f"Experiment created in {exp.logdir}")
        if args.run:
            run_experiment(args.as_run_args())
    except ExperimentAlreadyExistsException as e:
        if not args.overwrite:
            response = ""
            response = input(f"Experiment already exists in {e.logdir}. Overwrite? [y/n] ")
            if response.lower() != "y":
                logging.info("Experiment not created.")
                return
        shutil.rmtree(e.logdir)
        return main(args)


if __name__ == "__main__":
    tap.Parser(Arguments).bind(main).run()
