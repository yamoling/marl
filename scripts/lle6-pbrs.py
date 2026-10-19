import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal, get_args

import dotenv
import typed_argparse as tap

import marl
from marl import algos
from marl.algos.qtarget_updater import HardUpdate
from marl.env import LLEConfig
from marl.models import EpisodeMemory, Trainer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import CNNOptionCritic, actor_critics, qnetworks
from marl.policy import EpsilonGreedy
from marl.runners import parallel_run
from marl.utils import Schedule

logger = logging.getLogger(__name__)

Algo = Literal["dqn", "vdn", "qmix", "qplex", "mappo", "ippo", "maser", "haven", "option-critic"]
ALGOS: tuple[Algo, ...] = get_args(Algo)
GAMMA = 0.95
"""Discount factor of every algorithm. With gamma=0.99, VDN and MASER fail on this level (see the 2026-10-15 report)."""
MEMORY_SIZE = 50_000
HAVEN_N_SUBGOALS = 8
HAVEN_K = 3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_STEPS = 200_000
RLC2025_ALGOS: tuple[Algo, ...] = ("vdn", "qmix", "qplex", "maser", "haven")
"""Algorithms trained with the episodic setup of the RLC 2025 paper (pymarl configuration of ~/HAVEN-forked)."""
RLC2025_MEMORY_SIZE = 5_000
"""Number of episodes in the replay memory."""
RLC2025_EPSILON_STEPS = 50_000
RLC2025_TARGET_UPDATE_EPISODES = 200
RLC2025_BATCH_SIZE = 32
"""Number of episodes per batch."""
PPO_TRAIN_INTERVAL = 512
PPO_N_EPOCHS = 10
PPO_MINIBATCH_SIZE = 128
PPO_LR = 3e-4
PPO_ENTROPY_COEF = 0.05
IPPO_MLP_SIZES = (224,)
"""IPPO hyperparameters: best MAPPO trial (57) of `MAPPO-independent-9x9_agents3_lasers2` in
`tunings/certified-cooperation.journal`, with gamma=0.95. Selected by a 2M-step screening of 8 IPPO configurations."""
IPPO_TRAIN_INTERVAL = 150
IPPO_MINIBATCH_SIZE = 144
IPPO_N_EPOCHS = 16
IPPO_LR_ACTOR = 7.48e-5
IPPO_LR_CRITIC = 2.31e-5
IPPO_ENTROPY_COEF = 0.229
IPPO_EPS_CLIP = 0.177
IPPO_EARLY_STOPPING_KL = 0.0243
OC_N_OPTIONS = 3


def make_train_policy():
    """Fresh epsilon-greedy policy, linearly annealed over the first `EPSILON_STEPS` time steps. @ai-generated"""
    return EpsilonGreedy.linear(EPSILON_START, EPSILON_END, EPSILON_STEPS)


def make_env(extra_padding_size: int | None = None, rlc2025: bool = False, pbrs: bool = True):
    """
    Level 6 with PBRS on the laser crossings. The RLC 2025 setup gives each agent its own subgoal flags (instead of
    the joint flag matrix) and its last action as extras. Without `pbrs`, neither the shaping reward nor the subgoal
    flags are added (the baseline to compare PBRS with).

    @ai-edited
    """
    shaping = "individual" if rlc2025 else "joint"
    return LLEConfig(
        6,
        pbrs=shaping if pbrs else None,
        pbrs_subgoals="crossing",
        agent_id=True,
        last_action=rlc2025,
        extra_padding_size=extra_padding_size,
    )


def rlc2025_kwargs() -> dict[str, Any]:
    """
    Hyperparameters shared by the DQN-based trainers of the RLC 2025 paper: episode replay, RMSprop, double
    Q-learning, hard target updates every 200 episodes and a fast epsilon decay.

    @ai-generated
    """
    return {
        "train_policy": EpsilonGreedy.linear(EPSILON_START, EPSILON_END, RLC2025_EPSILON_STEPS),
        "gamma": GAMMA,
        "lr": 5e-4,
        "batch_size": RLC2025_BATCH_SIZE,
        "double_qlearning": True,
        "target_updater": HardUpdate(RLC2025_TARGET_UPDATE_EPISODES),
        "optimiser_type": "rmsprop",
        "grad_norm_clipping": 10.0,
        "train_interval": (1, "episode"),
    }


def make_haven(pbrs: bool = True):
    """Build HAVEN (VDN-mixed DQN at both levels) with the RLC 2025 setup and its padded environment. @ai-edited"""
    env = make_env(extra_padding_size=HAVEN_N_SUBGOALS, rlc2025=True, pbrs=pbrs)
    n_workers = env.n_agents
    n_meta_extras = env.extras_shape[0] - HAVEN_N_SUBGOALS
    n_agent_extras = 0
    meta_qnetwork = qnetworks.QCNN(
        HAVEN_N_SUBGOALS,
        n_workers,
        env.observation_shape,
        (n_meta_extras + HAVEN_N_SUBGOALS,),
        duelling=False,
    )
    worker_qnetwork = qnetworks.QCNN(
        env.n_actions,
        n_workers,
        env.observation_shape,
        (n_meta_extras + n_agent_extras + HAVEN_N_SUBGOALS,),
        duelling=False,
    )
    meta_trainer = algos.DQN(meta_qnetwork, EpisodeMemory(RLC2025_MEMORY_SIZE), mixer=mixers.VDN(), **rlc2025_kwargs())
    worker_trainer = algos.DQN(worker_qnetwork, EpisodeMemory(RLC2025_MEMORY_SIZE), mixer=mixers.VDN(), **rlc2025_kwargs())
    trainer = algos.HAVEN(
        meta_trainer,
        worker_trainer,
        n_workers,
        HAVEN_N_SUBGOALS,
        HAVEN_K,
        n_meta_extras,
        n_agent_extras,
        grad_norm_clipping=10.0,
        gamma=GAMMA,
    )
    return env, trainer


def make_env_and_trainer(algo: Algo, pbrs: bool = True) -> tuple[LLEConfig, Trainer]:
    """Build the environment (with or without PBRS) and the trainer of the given algorithm. @ai-edited"""
    if algo == "haven":
        return make_haven(pbrs)
    env = make_env(rlc2025=algo in RLC2025_ALGOS or algo in ("ippo", "option-critic"), pbrs=pbrs)
    match algo:
        case "dqn":
            return env, algos.DQN(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                train_policy=make_train_policy(),
                gamma=GAMMA,
                lr=5e-4,
                grad_norm_clipping=10.0,
            )
        case "vdn":
            return env, algos.VDN(
                qnetworks.from_env(env, duelling=False),
                EpisodeMemory(RLC2025_MEMORY_SIZE),
                **rlc2025_kwargs(),
            )
        case "qmix":
            return env, algos.QMix(
                qnetworks.from_env(env, duelling=False),
                EpisodeMemory(RLC2025_MEMORY_SIZE),
                mixer=mixers.QMix.from_env(env, embed_size=32),
                **rlc2025_kwargs(),
            )
        case "qplex":
            return env, algos.QPlex(
                qnetworks.from_env(env, duelling=False),
                EpisodeMemory(RLC2025_MEMORY_SIZE),
                mixer=mixers.QPlex.from_env(env),
                **rlc2025_kwargs(),
            )
        case "mappo":
            actor, critic = actor_critics.from_env(env, recurrent=False)
            return env, algos.PPO(
                actor,
                critic,
                mixers.VDN(),
                train_interval=(PPO_TRAIN_INTERVAL, "step"),
                n_epochs=PPO_N_EPOCHS,
                minibatch_size=PPO_MINIBATCH_SIZE,
                lr_actor=PPO_LR,
                lr_critic=PPO_LR,
                c1=Schedule.constant(1.0),
                c2=Schedule.constant(PPO_ENTROPY_COEF),
                gamma=GAMMA,
                grad_norm_clipping=10.0,
            )
        case "ippo":
            actor, critic = actor_critics.from_env(
                env,
                False,
                actor_kwargs={"mlp_sizes": IPPO_MLP_SIZES},
                critic_kwargs={"mlp_sizes": IPPO_MLP_SIZES},
            )
            return env, algos.PPO(
                actor,
                critic,
                None,
                train_interval=(IPPO_TRAIN_INTERVAL, "step"),
                n_epochs=IPPO_N_EPOCHS,
                minibatch_size=IPPO_MINIBATCH_SIZE,
                lr_actor=IPPO_LR_ACTOR,
                lr_critic=IPPO_LR_CRITIC,
                c1=Schedule.constant(1.0),
                c2=Schedule.constant(IPPO_ENTROPY_COEF),
                eps_clip=IPPO_EPS_CLIP,
                normalize_advantages=True,
                early_stopping_kl=IPPO_EARLY_STOPPING_KL,
                gamma=GAMMA,
            )
        case "option-critic":
            return env, algos.OptionCritic(
                CNNOptionCritic.from_env(env, n_options=OC_N_OPTIONS),
                env.n_agents,
                mixer=mixers.VDN.from_env(env),
                gamma=GAMMA,
                option_train_policy=EpsilonGreedy.linear(EPSILON_START, EPSILON_END, EPSILON_STEPS),
            )
        case "maser":
            return env, algos.MASER(
                qnetworks.from_env(env, duelling=False),
                EpisodeMemory(RLC2025_MEMORY_SIZE),
                mixer=mixers.VDN.from_env(env),
                alpha=0.5,
                intrinsic_weight=0.03,
                **rlc2025_kwargs(),
            )


class Args(tap.TypedArgs):
    algos: list[Algo] = tap.arg("--algos", nargs="+", default=list(ALGOS))
    n_seeds: int = tap.arg("--n-seeds", default=16)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    n_jobs: int = tap.arg("--n-jobs", default=1)
    n_tests: int = tap.arg("--n-tests", default=1)
    test_interval: int = tap.arg("--test-interval", default=5_000)
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False)
    no_pbrs: bool = tap.arg("--no-pbrs", default=False, help="Same hyperparameters without reward shaping.")

    @property
    def requested_seeds(self):
        return list(range(self.start_seed, self.start_seed + self.n_seeds))


def logdir_of(algo: Algo, n_steps: int, pbrs: bool = True):
    """
    Keep the RLC 2025 setups and the new MAPPO separate from experiments using the old defaults, and the runs
    without PBRS separate from the ones with PBRS.

    @ai-edited
    """
    if algo in RLC2025_ALGOS:
        suffix = "rlc2025"
    elif algo == "mappo":
        suffix = f"ppo{PPO_TRAIN_INTERVAL}x{PPO_N_EPOCHS}-ent{PPO_ENTROPY_COEF:g}"
    elif algo == "option-critic":
        suffix = f"{OC_N_OPTIONS}options-rlc2025env"
    elif algo == "ippo":
        suffix = f"ppo{IPPO_TRAIN_INTERVAL}x{IPPO_N_EPOCHS}-ent{IPPO_ENTROPY_COEF:g}-rlc2025env"
    else:
        suffix = f"eps{EPSILON_END:g}"
    if not pbrs:
        return Path("logs", f"lle6-nopbrs-{algo}-{n_steps}-{suffix}")
    return Path("logs", f"lle6-pbrs-{algo}-{n_steps}-crossing-{suffix}")


def get_experiment(args: Args, algo: Algo) -> marl.Experiment:
    logdir = logdir_of(algo, args.n_steps, not args.no_pbrs)
    if logdir.exists():
        return marl.Experiment.load(logdir)
    env, trainer = make_env_and_trainer(algo, not args.no_pbrs)
    exp = marl.Experiment.create(env, trainer, logdir=logdir, n_steps=args.n_steps)
    logger.info(f"Created experiment {exp.logdir}")
    return exp


def missing_seeds(args: Args, logdir: Path) -> list[int]:
    """
    Requested seeds without any run in the experiment. Existing runs (complete, running or interrupted)
    are never overwritten.

    @ai-generated
    """
    if not logdir.exists():
        return args.requested_seeds
    existing = {run.seed for run in marl.Experiment.load(logdir).runs}
    return [seed for seed in args.requested_seeds if seed not in existing]


def main(args: Args):
    """Create the experiments and train their missing seeds in one shared parallel queue. @ai-generated"""
    if args.n_seeds <= 0:
        raise ValueError(f"--n-seeds must be positive, got {args.n_seeds}")
    if args.n_jobs <= 0:
        raise ValueError(f"--n-jobs must be positive, got {args.n_jobs}")
    runs = list[marl.Run]()
    for algo in args.algos:
        logdir = logdir_of(algo, args.n_steps, not args.no_pbrs)
        seeds = missing_seeds(args, logdir)
        if len(seeds) == 0:
            logger.info(f"[skip] All requested seeds already exist in {logdir}")
            continue
        if args.dry_run:
            status = "exists" if logdir.exists() else "new"
            logger.info(f"[{status}] {len(seeds)} runs {seeds} -> {logdir}")
            continue
        exp = get_experiment(args, algo)
        logger.info(f"Queueing {exp.logdir} with seeds {seeds}")
        runs += exp.create_runs(
            seeds,
            n_tests=args.n_tests,
            test_interval=args.test_interval,
            save_weights=False,
            save_actions=False,
            save_training_data=False,
        )
    if len(runs) == 0:
        return
    logger.info(f"Starting {len(runs)} runs with {args.n_jobs} parallel jobs")
    parallel_run(
        runs,
        n_jobs=args.n_jobs,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=args.quiet,
        limit_torch_threads=None,
    )


if __name__ == "__main__":
    # Episode batches have variable lengths: without expandable segments, the CUDA caching allocator of the
    # workers fragments and reserves 5-10 GB per run instead of ~2 GB. Inherited by the worker processes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("lle6_pbrs.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception(f"The LLE level-6 PBRS study failed with command line '{sys.argv}'.")
        raise
