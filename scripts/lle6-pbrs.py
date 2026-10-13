import logging
import os
import sys
from pathlib import Path
from typing import Literal, get_args

import dotenv
import typed_argparse as tap

import marl
from marl import algos
from marl.env import LLEConfig
from marl.models import EpisodeMemory, Trainer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.policy import EpsilonGreedy
from marl.runners import parallel_run

logger = logging.getLogger(__name__)

Algo = Literal["dqn", "vdn", "qmix", "mappo", "maser", "haven"]
ALGOS: tuple[Algo, ...] = get_args(Algo)
MEMORY_SIZE = 50_000
MASER_MEMORY_SIZE = 5_000
"""Number of episodes in MASER's buffer (as in the paper)."""
HAVEN_N_SUBGOALS = 8
HAVEN_K = 5
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_STEPS = 200_000


def make_train_policy():
    """Fresh epsilon-greedy policy, linearly annealed over the first `EPSILON_STEPS` time steps. @ai-generated"""
    return EpsilonGreedy.linear(EPSILON_START, EPSILON_END, EPSILON_STEPS)


def make_env(extra_padding_size: int | None = None):
    return LLEConfig(6, pbrs="joint", pbrs_subgoals="crossing", agent_id=True, extra_padding_size=extra_padding_size)


def make_haven():
    """Build HAVEN (VDN-mixed DQN at both levels) and its padded environment. @ai-generated"""
    env = make_env(extra_padding_size=HAVEN_N_SUBGOALS)
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
    meta_trainer = algos.DQN(meta_qnetwork, TransitionMemory(MEMORY_SIZE // HAVEN_K), mixer=mixers.VDN(), train_policy=make_train_policy())
    worker_trainer = algos.DQN(worker_qnetwork, TransitionMemory(MEMORY_SIZE), mixer=mixers.VDN(), train_policy=make_train_policy())
    trainer = algos.HAVEN(meta_trainer, worker_trainer, n_workers, HAVEN_N_SUBGOALS, HAVEN_K, n_meta_extras, n_agent_extras)
    return env, trainer


def make_env_and_trainer(algo: Algo) -> tuple[LLEConfig, Trainer]:
    """Build the PBRS environment and the default trainer of the given algorithm. @ai-generated"""
    if algo == "haven":
        return make_haven()
    env = make_env()
    match algo:
        case "dqn":
            return env, algos.DQN(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                train_policy=make_train_policy(),
                gamma=0.95,
                lr=5e-4,
                grad_norm_clipping=10.0,
            )
        case "vdn":
            # Hyperparameters of the ECAI 2024 paper.
            return env, algos.VDN(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                train_policy=make_train_policy(),
                gamma=0.95,
                lr=5e-4,
                grad_norm_clipping=10.0,
            )
        case "qmix":
            return env, algos.QMix(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                mixer=mixers.QMix.from_env(env),
                train_policy=make_train_policy(),
                gamma=0.95,
                lr=5e-4,
            )
        case "mappo":
            actor, critic = actor_critics.from_env(env, recurrent=False)
            return env, algos.PPO(actor, critic, mixers.VDN(), gamma=0.95)
        case "maser":
            memory = EpisodeMemory(MASER_MEMORY_SIZE)
            return env, algos.MASER(
                qnetworks.from_env(env),
                memory,
                mixer=mixers.VDN(),
                train_interval=(1, "episode"),
                train_policy=make_train_policy(),
                gamma=0.95,
                lr=5e-4,
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

    @property
    def requested_seeds(self):
        return list(range(self.start_seed, self.start_seed + self.n_seeds))


def logdir_of(algo: Algo, n_steps: int):
    return Path("logs", f"lle6-pbrs-{algo}-{n_steps}-crossing-eps{EPSILON_END:g}")


def get_experiment(args: Args, algo: Algo) -> marl.Experiment:
    """Load the experiment of `algo` if it exists, otherwise create it. @ai-generated"""
    logdir = logdir_of(algo, args.n_steps)
    if logdir.exists():
        return marl.Experiment.load(logdir)
    env, trainer = make_env_and_trainer(algo)
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
        logdir = logdir_of(algo, args.n_steps)
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
            seeds, n_tests=args.n_tests, test_interval=args.test_interval, save_weights=False, save_actions=False, save_training_data=False
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
