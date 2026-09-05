"""Reproduce the SMAC experiment of the LAIES paper (Liu et al., ICML 2023).

LAIES ("Lazy Agents: A New Perspective on Solving Sparse Reward Problem in Multi-agent Reinforcement
Learning", https://proceedings.mlr.press/v202/liu23ac.html) is QMIX augmented with two intrinsic rewards
computed from an external-state transition model: individual diligence (IDI) and collaborative diligence
(CDI). The paper's central claim is that, on the *sparse-reward* version of SMAC (+1 win, -1 defeat, 0
otherwise), QMIX gets stuck at a 0% win rate because agents become lazy, whereas LAIES learns a winning
policy that even matches or beats QMIX trained with SMAC's hand-crafted dense reward ("QMIX-DR").

This script trains the three conditions of Figure 4 of the paper:

- `laies`   : LAIES on the sparse-reward map, with the intrinsic rewards annealed once the mean return
              becomes positive, as Appendix A.1 prescribes;
- `laies-noanneal` : the same, but keeping the intrinsic rewards active for the whole run. The paper does
              not report this ablation; it separates the contribution of the diligence rewards from the
              contribution of the annealing schedule;
- `qmix`    : vanilla QMIX on the sparse-reward map (expected to stay at 0% win rate);
- `qmix-dr` : vanilla QMIX on the default dense-reward map (the paper's strong reference).

All three share the QMIX hyperparameters of PyMARL (recurrent agents with shared parameters, RMSProp at
5e-4, batches of 32 episodes, a 5,000-episode replay buffer, a hard target update every 200 episodes,
gamma = 0.99, gradient clipping at 10). The LAIES-specific beta_1 / beta_2 come from Appendix A.1 of the
paper.

Usage:
    SC2PATH=/path/to/StarCraftII uv run --extra=legacy-gpu python scripts/train_laies_smac.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import dotenv
import typed_argparse as tap

import marl
from marl.algos import DQN, LAIES, HardUpdate
from marl.env import SMACConfig
from marl.models import Trainer
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import ArgMax, EpsilonGreedy

LOGGER = logging.getLogger(__name__)

Condition = Literal["laies", "laies-noanneal", "qmix", "qmix-dr"]

# beta_1 (IDI) and beta_2 (CDI) reported in Appendix A.1 of the paper.
PAPER_BETAS: dict[str, tuple[float, float]] = {
    "3m": (100.0, 0.2),
    "1c3s5z": (100.0, 0.2),
    "3s_vs_3z": (100.0, 0.2),
    "8m_vs_9m": (100.0, 0.2),
    "MMM2": (100.0, 0.2),
    "6h_vs_8z": (100.0, 0.2),
    "2m_vs_1z": (600.0, 0.3),
    "5m_vs_6m": (200.0, 0.3),
    "MMM": (20.0, 0.02),
}


class Args(tap.TypedArgs):
    map_name: str = tap.arg("--map", default="2m_vs_1z", help="SMAC map to train on.")
    conditions: list[str] = tap.arg(
        "--conditions",
        nargs="+",
        default=["laies", "laies-noanneal", "qmix", "qmix-dr"],
        help="Which of the paper's curves to reproduce.",
    )
    logdir: str = tap.arg("--logdir", default="laies-paper", help="Root experiment directory, under logs/.")
    n_steps: int = tap.arg("--n-steps", default=500_000, help="Training steps per run (0.5M for 2m_vs_1z).")
    n_seeds: int = tap.arg("--n-seeds", default=6, help="Number of seeds; the paper uses 6.")
    start_seed: int = tap.arg("--start-seed", default=0)
    n_jobs: int = tap.arg("--n-jobs", default=18, help="Runs to train in parallel (all conditions at once).")
    gpus: list[int] = tap.arg("--gpus", nargs="+", default=list(range(8)))
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    test_interval: int = tap.arg("--test-interval", default=10_000)
    n_tests: int = tap.arg("--n-tests", default=32, help="Test episodes per evaluation point.")
    epsilon_anneal: int = tap.arg("--epsilon-anneal", default=50_000, help="Steps to anneal epsilon 1.0 -> 0.05.")
    beta_idi: float | None = tap.arg("--beta-idi", default=None, help="Overrides the paper's beta_1.")
    beta_cdi: float | None = tap.arg("--beta-cdi", default=None, help="Overrides the paper's beta_2.")
    cdi_samples: int = tap.arg("--cdi-samples", default=4, help="Joint counterfactual actions sampled for CDI.")
    intrinsic_anneal: int = tap.arg(
        "--intrinsic-anneal",
        default=100_000,
        help="Steps over which intrinsic rewards decay once the mean return becomes positive. 0 disables it.",
    )
    quiet: bool = tap.arg("--quiet", default=True)
    resume: bool = tap.arg("--resume", default=False, help="Only train the seeds that are missing.")
    sc2path: str = tap.arg("--sc2path", default="/home/yamoling/3rdparty/StarCraftII")


def enemy_state_indices(map_name: str) -> tuple[int, ...]:
    """
    Return the indices of the enemy features inside SMAC's flat global state.

    The paper defines the external states of SMAC as "opponents' positions and health". SMAC's state
    concatenates, in this order, `n_agents` ally blocks of `4 + shield_bits_ally + unit_type_bits`
    features, `n_enemies` enemy blocks of `3 + shield_bits_enemy + unit_type_bits` features
    (health, relative x, relative y, shield, unit type), the last actions and, optionally, the timestep.
    We therefore keep every enemy feature except the constant unit-type one-hot.

    @ai-generated
    """
    from smac.env import StarCraft2Env  # pyright: ignore[reportMissingImports]

    # Building a StarCraft2Env does not launch the game: it only reads the map parameters.
    env = StarCraft2Env(map_name=map_name)
    n_ally_features = 4 + env.shield_bits_ally + env.unit_type_bits
    n_enemy_features = 3 + env.shield_bits_enemy + env.unit_type_bits
    offset = env.n_agents * n_ally_features
    indices = []
    for enemy in range(env.n_enemies):
        start = offset + enemy * n_enemy_features
        indices.extend(range(start, start + n_enemy_features - env.unit_type_bits))
    return tuple(indices)


def make_env(map_name: str, dense: bool) -> SMACConfig:
    """
    Build the SMAC configuration of the paper.

    With `reward_sparse=True`, SMAC still divides the reward by `max_reward / reward_scale_rate` unless
    `reward_scale` is disabled, so both flags are needed to obtain exactly the +1 / -1 / 0 reward that the
    paper describes. The dense configuration is SMAC's default shaped reward, which is what QMIX-DR uses.

    @ai-generated
    """
    if dense:
        return SMACConfig(map_name, reward_sparse=False, reward_scale=True)
    return SMACConfig(map_name, reward_sparse=True, reward_scale=False)


def make_trainer(condition: Condition, env: SMACConfig, args: Args) -> Trainer:
    """
    Build the QMIX or LAIES trainer with the hyperparameters of the paper.

    @ai-generated
    """
    common = {
        "qnetwork": qnetworks.from_env(
            env, recurrent=True, duelling=False, independent=False, mlp_head_sizes=(64,), mlp_tail_sizes=(64,)
        ),
        "mixer": mixers.QMix.from_env(env),
        "memory_size": 5_000,
        "batch_size": 32,
        "train_interval": (1, "episode"),
        "train_policy": EpsilonGreedy.linear(1.0, 0.05, args.epsilon_anneal),
        "test_policy": ArgMax(),
        "target_updater": HardUpdate(200),
        "optimiser_type": "rmsprop",
        "lr": 5e-4,
        "double_qlearning": True,
        "gamma": 0.99,
        "grad_norm_clipping": 10.0,
    }
    if not condition.startswith("laies"):
        return DQN(**common)  # type: ignore[arg-type]
    beta_idi, beta_cdi = PAPER_BETAS.get(env.map_name, (1.0, 1.0))
    if args.beta_idi is not None:
        beta_idi = args.beta_idi
    if args.beta_cdi is not None:
        beta_cdi = args.beta_cdi
    return LAIES(
        external_state_indices=enemy_state_indices(env.map_name),
        beta_idi=beta_idi,
        beta_cdi=beta_cdi,
        cdi_samples=args.cdi_samples,
        estm_hidden_size=128,
        estm_lr=3e-4,
        # The paper does not clip the intrinsic reward: beta_1 is up to 600, so clipping at 1 would
        # saturate the signal and destroy the ranking between more and less diligent joint actions.
        intrinsic_reward_clip=None,
        intrinsic_anneal_steps=args.intrinsic_anneal if condition == "laies" else 0,
        **common,  # type: ignore[arg-type]
    )


def build_experiment(condition: Condition, args: Args):
    """
    Load or create the experiment of one condition and return it together with the seeds left to train.

    @ai-generated
    """
    logpath = Path("logs") / args.logdir / args.map_name / condition
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    if logpath.exists():
        if not args.resume:
            raise FileExistsError(f"{logpath} already exists. Use --resume to train the missing seeds.")
        experiment = marl.Experiment.load(logpath)
        complete = {run.seed for run in experiment.runs if run.is_complete}
        seeds = [seed for seed in seeds if seed not in complete]
    else:
        env = make_env(args.map_name, dense=condition == "qmix-dr")
        trainer = make_trainer(condition, env, args)
        experiment = marl.Experiment.create(env, trainer, logdir=logpath, n_steps=args.n_steps)
        LOGGER.info("Created %s (%s) on %s", logpath, trainer.name, experiment.env.name)
    return experiment, seeds


def main(args: Args):
    """
    Create every condition of the experiment and train all their runs in a single parallel pool.

    @ai-generated
    """
    from marl.runners import parallel_run

    os.environ.setdefault("SC2PATH", args.sc2path)
    runs = []
    for condition in args.conditions:
        if condition not in ("laies", "laies-noanneal", "qmix", "qmix-dr"):
            raise ValueError(f"Unknown condition {condition}")
        experiment, seeds = build_experiment(condition, args)  # type: ignore[arg-type]
        if not seeds:
            LOGGER.info("All seeds of %s are already complete.", experiment.logdir)
            continue
        runs.extend(
            experiment.create_runs(
                seeds,
                n_tests=args.n_tests,
                test_interval=args.test_interval,
                save_weights=True,
                save_actions=False,
            )
        )
    if not runs:
        LOGGER.info("Nothing to train.")
        return
    LOGGER.info("Training %d runs with %d parallel jobs.", len(runs), args.n_jobs)
    parallel_run(
        runs,
        n_jobs=min(args.n_jobs, len(runs)),
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=[gpu for gpu in range(8) if gpu not in args.gpus],
        quiet=args.quiet,
        limit_torch_threads=True,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    logging.basicConfig(
        handlers=[logging.FileHandler("train_laies_smac.log", mode="a"), logging.StreamHandler()],
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as error:
        LOGGER.error("LAIES SMAC command failed: %s", " ".join(sys.argv), exc_info=error)
        raise
