"""
Run MAPPO, IPPO, DQN (independent, no mixer), VDN and QMIX on LLE levels 1 to 5 (included) with
layered observations, for 1M time steps and 10 seeds.

- MAPPO/IPPO use the actor/critic networks from `actor_critics.from_env` and the `PPO` arguments
  set in `make_trainer`. MAPPO uses a VDN (sum) mixer over the individual critic values.
- DQN/VDN/QMIX use the default hyperparameters of their trainer class, the default Q-network from
  `qnetworks.from_env`, and a `TransitionMemory` of `MEMORY_SIZE` transitions (the memory has no
  default). QMIX uses `mixers.QMix.from_env` with its default sizes.

All runs of all experiments share a single job queue. Existing experiments are reused and only
their missing seeds are launched, so the script can safely be re-run to complete a sweep.

Run with `uv run python scripts/run_ppo_lle1to5.py [options]` from the repository root.

@ai-generated
"""

import logging
from pathlib import Path
from typing import Literal, cast

import torch
import typed_argparse as tap

import marl
from marl import Experiment, algos
from marl.env import LLEConfig
from marl.models import Run, Trainer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.runners import parallel_run
from marl.utils import DeviceLike

logger = logging.getLogger(__name__)

LEVELS = [1, 2, 3, 4, 5]
Algo = Literal["mappo", "ippo", "dqn", "vdn", "qmix"]
ALGOS: tuple[Algo, ...] = ("mappo", "ippo", "dqn", "vdn", "qmix")
N_STEPS = 1_000_000
MEMORY_SIZE = 50_000
LR = 5e-4
GRAD_NORM_CLIP = 10.0


def make_trainer(env: LLEConfig, algo: Algo) -> Trainer:
    e_greedy = marl.policy.EpsilonGreedy.linear(1, 0.05, 200_000)
    match algo:
        case "mappo" | "ippo":
            actor, critic = actor_critics.from_env(env, recurrent=False)
            mixer = mixers.VDN() if algo == "mappo" else None
            return algos.PPO(actor, critic, mixer, minibatch_size=32, n_epochs=10, early_stopping_kl=0.1)
        case "dqn":
            return algos.DQN(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                lr=LR,
                grad_norm_clipping=GRAD_NORM_CLIP,
                train_policy=e_greedy,
            )
        case "vdn":
            return algos.VDN(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                lr=LR,
                grad_norm_clipping=GRAD_NORM_CLIP,
                train_policy=e_greedy,
            )
        case "qmix":
            return algos.QMix(
                qnetworks.from_env(env),
                TransitionMemory(MEMORY_SIZE),
                mixer=mixers.QMix.from_env(env),
                lr=LR,
                grad_norm_clipping=GRAD_NORM_CLIP,
                train_policy=e_greedy,
            )


class Args(tap.TypedArgs):
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only print what would be created/run.")
    levels: list[int] = tap.arg("--levels", default=list(LEVELS), nargs="*", help=f"LLE levels to run (default: all, {list(LEVELS)}).")
    n_seeds: int = tap.arg("--n-seeds", default=10, help="Number of seeds per experiment (seeds 0..n-1).")
    device: DeviceLike = tap.arg("--device", default="auto", help='"auto", "cpu", "cuda:<id>" or a GPU index.')
    n_jobs: int | None = tap.arg("--n-jobs", default=None, help="Number of parallel run processes (defaults to the number of usable GPUs).")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    test_interval: int = tap.arg("--test-interval", default=10_000)
    n_tests: int = tap.arg("--n-tests", default=5)
    quiet: bool = tap.arg("--quiet", default=False)


def collect_runs(args: Args) -> list[Run]:
    """
    Create the experiments that do not exist yet and the runs of all seeds that are missing.
    """
    invalid = set(args.levels) - set(LEVELS)
    if invalid:
        raise ValueError(f"Invalid level(s) {sorted(invalid)}; available levels are {list(LEVELS)}.")
    runs = []
    for level in cast(list[Literal[1, 2, 3, 4, 5]], args.levels):
        for algo in ALGOS:
            name = f"lle{level}-{algo}"
            logdir = Path("logs") / name
            if logdir.exists():
                exp = Experiment.load(logdir)
                existing = {run.seed for run in exp.runs}
            else:
                exp = None
                existing = set()
            seeds = [s for s in range(args.n_seeds) if s not in existing]
            if not seeds:
                logger.info(f"[skip] {name}: all seeds already exist.")
                continue
            if args.dry_run:
                logger.info(f"[dry-run] {name}: would run seeds {seeds} -> {logdir}")
                continue
            if exp is None:
                env = LLEConfig(level, obs_type="layered")
                exp = Experiment.create(env, make_trainer(env, algo), logdir=name, n_steps=N_STEPS)
                logger.info(f"[new] Created experiment {exp.logdir}")
            logger.info(f"{name}: queueing seeds {seeds}")
            runs += exp.create_runs(seeds, args.n_tests, args.test_interval, save_weights=True, save_actions=True)
    return runs


def main(args: Args):
    runs = collect_runs(args)
    if not runs:
        return
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = 1 if args.device == "cpu" else max(1, torch.cuda.device_count() - len(args.disabled_gpus))
    logger.info(f"Launching {len(runs)} run(s) with {n_jobs} parallel job(s).")
    parallel_run(
        runs,
        n_jobs=n_jobs,
        device=args.device,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tap.Parser(Args).bind(main).run()
