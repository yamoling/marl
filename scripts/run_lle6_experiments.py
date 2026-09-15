"""
Reproduce the LLE level-6 experiments described in `.agents/plans/experiment.md`, with the
deviations requested by the user:

- IQL (i.e. `DQN` with `mixer=None`), VDN and QMIX baselines, 20 seeds, no extensions.
- MAPPO and IPPO using the hyperparameters commonly reported in the papers that introduced
  them (Yu et al., 2022 for MAPPO; de Witt et al., 2020 for IPPO), without a grid search and
  without any extension.
- PER, n-step returns and RND applied on top of IQL, VDN and QMIX (the plan document only
  covers VDN/QMIX, this script widens it to IQL as well), using the "best" values linked in
  the plan document.
- ICM applied on top of VDN and QMIX only (matching the plan document's scope), using the
  paper's own hyperparameters (Pathak et al., 2017) rather than a grid search.
- Q-networks and actor/critic networks are built exclusively through `qnetworks.from_env` /
  `actor_critics.from_env` with their defaults; no attempt is made to reproduce the exact
  architecture table in the plan document.
- `state_type="flattened"` for the LLE environment (this is actually `LLEConfig`'s default).

Run with `uv run python scripts/run_lle6_experiments.py [options]` from the repository root.

@ai-generated
"""

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dotenv
import typed_argparse as tap

from marl import Experiment, algos
from marl.env import LLEConfig
from marl.models import NStepMemory, PrioritizedMemory, ReplayMemory, Trainer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.policy import EpsilonGreedy
from marl.utils import DeviceLike, Schedule

logger = logging.getLogger(__name__)

LLE_LEVEL: Literal[6] = 6
N_STEPS = 1_000_000

# "Main" hyperparameters table (IQL/VDN/QMIX and their PER/n-step/RND extensions).
MEMORY_SIZE = 50_000
BATCH_SIZE = 64
TRAIN_INTERVAL: tuple[int, Literal["step"]] = (5, "step")
LR = 5e-4
GRAD_NORM_CLIPPING = 10.0
GAMMA = 0.95
TAU = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_STEPS = 200_000

# PER hyperparameters (best values from the grid search referenced in the plan document).
PER_ALPHA = Schedule.constant(0.6)
PER_BETA_START = 0.5
PER_BETA_END = 1.0
PER_BETA = Schedule.linear(PER_BETA_START, PER_BETA_END, N_STEPS)
PER_TD_ERROR_CLIPPING = 5.0

# n-step hyperparameters (best value from the grid search referenced in the plan document).
NSTEP_N = 3

# RND hyperparameters (best values from the grid search referenced in the plan document).
RND_UPDATE_RATIO = 0.25
RND_MU_START = 2.0
RND_MU_END = 2 / 3

# MAPPO/IPPO hyperparameters, following the values commonly reported by Yu et al. (2022) for
# MAPPO on SMAC (reused unchanged for IPPO, as de Witt et al. (2020) reuse the same on-policy
# codebase and hyperparameters with little additional tuning). `gamma` is kept at 0.95 to stay
# comparable with the value-based baselines, as suggested by the plan document.
PPO_TRAIN_INTERVAL: tuple[int, Literal["step"]] = (128, "step")
PPO_LR = 5e-4
PPO_N_EPOCHS = 15
PPO_EPS_CLIP = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_GAMMA = 0.95
PPO_VALUE_LOSS_COEF = 1.0
PPO_ENTROPY_COEF = 0.01
PPO_GRAD_NORM_CLIPPING = 10.0
PPO_MINIBATCH_SIZE = PPO_TRAIN_INTERVAL[0]  # A single minibatch per epoch, as in the paper.


def make_env() -> LLEConfig:
    """LLE level 6, layered observations, flattened state (the `LLEConfig` default). @ai-generated"""
    return LLEConfig(LLE_LEVEL, obs_type="layered", state_type="flattened")


def make_plain_memory() -> TransitionMemory:
    return TransitionMemory(MEMORY_SIZE)


def make_per_memory() -> PrioritizedMemory:
    """@ai-generated"""
    return PrioritizedMemory(
        TransitionMemory(MEMORY_SIZE),
        multi_objective=False,
        alpha=PER_ALPHA,
        beta=PER_BETA,
        td_error_clipping=PER_TD_ERROR_CLIPPING,
    )


def make_nstep_memory() -> NStepMemory:
    return NStepMemory(MEMORY_SIZE, n=NSTEP_N, gamma=GAMMA)


def make_rnd(env: LLEConfig) -> algos.RND:
    """@ai-generated"""
    return algos.RND(
        env.state_shape,
        env.state_extras_size,
        update_ratio=RND_UPDATE_RATIO,
        ir_weight=Schedule.linear(RND_MU_START, RND_MU_END, N_STEPS),
    )


def make_icm(env: LLEConfig) -> algos.ICM:
    """
    Pathak et al. (2017) defaults, which also happen to be `ICM`'s own defaults: `n_features=256`,
    `hidden_size=256`, `beta=0.2`, `lr=1e-3`, `grad_norm_clipping=40`, `weight=Schedule.constant(0.01)`.

    @ai-generated
    """
    return algos.ICM.from_env(env)


def make_value_trainer(
    env: LLEConfig,
    algo: Literal["iql", "vdn", "qmix"],
    memory: ReplayMemory,
    ir_module: algos.RND | algos.ICM | None = None,
) -> Trainer:
    """Build an IQL/VDN/QMIX trainer from the "Main" hyperparameters table. @ai-generated"""
    qnetwork = qnetworks.from_env(env)
    train_policy = EpsilonGreedy.linear(EPSILON_START, EPSILON_END, EPSILON_STEPS)
    target_updater = algos.SoftUpdate(TAU)
    match algo:
        case "iql":
            return algos.DQN(
                qnetwork,
                mixer=None,
                memory=memory,
                train_policy=train_policy,
                lr=LR,
                batch_size=BATCH_SIZE,
                train_interval=TRAIN_INTERVAL,
                grad_norm_clipping=GRAD_NORM_CLIPPING,
                gamma=GAMMA,
                target_updater=target_updater,
                optimiser_type="adam",
                double_qlearning=True,
                ir_module=ir_module,
            )
        case "vdn":
            return algos.VDN(
                qnetwork,
                memory=memory,
                train_policy=train_policy,
                lr=LR,
                batch_size=BATCH_SIZE,
                train_interval=TRAIN_INTERVAL,
                grad_norm_clipping=GRAD_NORM_CLIPPING,
                gamma=GAMMA,
                target_updater=target_updater,
                optimiser_type="adam",
                double_qlearning=True,
                ir_module=ir_module,
            )
        case "qmix":
            return algos.QMix(
                qnetwork,
                mixer=mixers.QMix.from_env(env),
                memory=memory,
                train_policy=train_policy,
                lr=LR,
                batch_size=BATCH_SIZE,
                train_interval=TRAIN_INTERVAL,
                grad_norm_clipping=GRAD_NORM_CLIPPING,
                gamma=GAMMA,
                target_updater=target_updater,
                optimiser_type="adam",
                double_qlearning=True,
                ir_module=ir_module,
            )


def make_ppo_trainer(env: LLEConfig, mappo: bool) -> algos.PPO:
    """Build a MAPPO (mixer=VDN sum) or IPPO (no mixer) trainer. @ai-generated"""
    actor, critic = actor_critics.from_env(env, recurrent=False)
    mixer = mixers.VDN() if mappo else None
    return algos.PPO(
        actor,
        critic,
        mixer,
        train_interval=PPO_TRAIN_INTERVAL,
        lr_actor=PPO_LR,
        lr_critic=PPO_LR,
        n_epochs=PPO_N_EPOCHS,
        eps_clip=PPO_EPS_CLIP,
        c1=Schedule.constant(PPO_VALUE_LOSS_COEF),
        c2=Schedule.constant(PPO_ENTROPY_COEF),
        gae_lambda=PPO_GAE_LAMBDA,
        minibatch_size=PPO_MINIBATCH_SIZE,
        grad_norm_clipping=PPO_GRAD_NORM_CLIPPING,
        gamma=PPO_GAMMA,
    )


@dataclass(frozen=True)
class RunSpec:
    name: str
    """Experiment log directory name, rooted under `logs/`."""
    build: Callable[[], tuple[LLEConfig, Trainer]]
    n_steps: int = N_STEPS


def _build_iql():
    env = make_env()
    return env, make_value_trainer(env, "iql", make_plain_memory())


def _build_vdn():
    env = make_env()
    return env, make_value_trainer(env, "vdn", make_plain_memory())


def _build_qmix():
    env = make_env()
    return env, make_value_trainer(env, "qmix", make_plain_memory())


def _build_iql_per():
    env = make_env()
    return env, make_value_trainer(env, "iql", make_per_memory())


def _build_iql_nstep():
    env = make_env()
    return env, make_value_trainer(env, "iql", make_nstep_memory())


def _build_iql_rnd():
    env = make_env()
    return env, make_value_trainer(env, "iql", make_plain_memory(), ir_module=make_rnd(env))


def _build_vdn_per():
    env = make_env()
    return env, make_value_trainer(env, "vdn", make_per_memory())


def _build_vdn_nstep():
    env = make_env()
    return env, make_value_trainer(env, "vdn", make_nstep_memory())


def _build_vdn_rnd():
    env = make_env()
    return env, make_value_trainer(env, "vdn", make_plain_memory(), ir_module=make_rnd(env))


def _build_vdn_icm():
    env = make_env()
    return env, make_value_trainer(env, "vdn", make_plain_memory(), ir_module=make_icm(env))


def _build_qmix_per():
    env = make_env()
    return env, make_value_trainer(env, "qmix", make_per_memory())


def _build_qmix_nstep():
    env = make_env()
    return env, make_value_trainer(env, "qmix", make_nstep_memory())


def _build_qmix_rnd():
    env = make_env()
    return env, make_value_trainer(env, "qmix", make_plain_memory(), ir_module=make_rnd(env))


def _build_qmix_icm():
    env = make_env()
    return env, make_value_trainer(env, "qmix", make_plain_memory(), ir_module=make_icm(env))


def _build_mappo():
    env = make_env()
    return env, make_ppo_trainer(env, mappo=True)


def _build_ippo():
    env = make_env()
    return env, make_ppo_trainer(env, mappo=False)


RUN_SPECS: list[RunSpec] = [
    RunSpec("lle6-iql", _build_iql),
    RunSpec("lle6-vdn", _build_vdn),
    RunSpec("lle6-qmix", _build_qmix),
    RunSpec("lle6-iql-per", _build_iql_per),
    RunSpec("lle6-iql-nstep3", _build_iql_nstep),
    RunSpec("lle6-iql-rnd", _build_iql_rnd),
    RunSpec("lle6-vdn-per", _build_vdn_per),
    RunSpec("lle6-vdn-nstep3", _build_vdn_nstep),
    RunSpec("lle6-vdn-rnd", _build_vdn_rnd),
    RunSpec("lle6-vdn-icm", _build_vdn_icm),
    RunSpec("lle6-qmix-per", _build_qmix_per),
    RunSpec("lle6-qmix-nstep3", _build_qmix_nstep),
    RunSpec("lle6-qmix-rnd", _build_qmix_rnd),
    RunSpec("lle6-qmix-icm", _build_qmix_icm),
    RunSpec("lle6-mappo", _build_mappo),
    RunSpec("lle6-ippo", _build_ippo),
]


class Args(tap.TypedArgs):
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only print what would be created/run.")
    start_seed: int = tap.arg("--start-seed", default=0, help="First seed to run.")
    n_seeds: int = tap.arg("--n-seeds", default=10, help="Number of seeds to run, starting at --start-seed.")
    device: DeviceLike = tap.arg("--device", default="auto", help='Device to use ("auto", "cpu", "cuda:<gpu_id>", or an int index).')
    only: list[str] = tap.arg(
        "--only",
        default=[],
        nargs="*",
        help="If given, only run experiments whose name contains one of these substrings (e.g. --only vdn qmix).",
    )
    n_jobs: int | None = tap.arg("--n-jobs", default=None, help='Number of parallel run processes (defaults to "auto").')
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="group")
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    test_interval: int = tap.arg("--test-interval", default=5_000)
    n_tests: int = tap.arg("--n-tests", default=5)
    quiet: bool = tap.arg("--quiet", default=False, help="Disable progress bars and console logs during training.")

    @property
    def requested_seeds(self):
        return range(self.start_seed, self.start_seed + self.n_seeds)


def run_one(args: Args, spec: RunSpec):
    """
    Create (if needed) and run the missing seeds of one experiment, never overwriting an
    existing experiment directory or run.

    @ai-generated
    """
    logdir = Path("logs") / spec.name
    requested_seeds = args.requested_seeds
    if logdir.exists():
        exp = Experiment.load(logdir)
        existing_seeds = {run.seed for run in exp.runs}
        seeds = [s for s in requested_seeds if s not in existing_seeds]
        if not seeds:
            logger.info(f"[skip] {spec.name}: all requested seeds {list(requested_seeds)} already exist.")
            return
        if args.dry_run:
            logger.info(f"[dry-run][existing] {spec.name}: would add seeds {seeds} -> {logdir}")
            return
        logger.info(f"[existing] {spec.name}: adding seeds {seeds} -> {logdir}")
    else:
        seeds = list(requested_seeds)
        if args.dry_run:
            logger.info(f"[dry-run][new] {spec.name}: would create experiment and run seeds {seeds} -> {logdir}")
            return
        env, trainer = spec.build()
        exp = Experiment.create(env, trainer, logdir=spec.name, n_steps=spec.n_steps)
        logger.info(f"[new] Created experiment {exp.logdir}")
    if args.device == "cpu":
        limit = 1
    else:
        limit = "auto"
    exp.run(
        seeds=seeds,
        device=args.device,
        gpu_strategy=args.gpu_strategy,
        n_jobs="auto" if args.n_jobs is None else args.n_jobs,
        disabled_gpus=args.disabled_gpus,
        test_interval=args.test_interval,
        n_tests=args.n_tests,
        quiet=args.quiet,
        limit_torch_threads=limit,
    )


def main(args: Args):
    specs = RUN_SPECS
    if args.only:
        specs = [s for s in specs if any(token in s.name for token in args.only)]
        if not specs:
            raise ValueError(f"--only {args.only} matched no experiment name. Available: {[s.name for s in RUN_SPECS]}")
    logger.info(f"Running {len(specs)}/{len(RUN_SPECS)} experiment(s) with seeds {list(args.requested_seeds)}.")
    for spec in specs:
        run_one(args, spec)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("run_lle6_experiments.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception(f"An error occurred while running the LLE level-6 experiments with command line '{sys.argv}'")
        raise
