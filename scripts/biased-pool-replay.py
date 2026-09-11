"""
Does a replay memory biased towards optimal demonstrations help an algorithm generalise over a
pool of layouts?

Experimental design
-------------------
The pool directory is split into a *training* pool (the first `--pool-size` layouts) and a
*held-out* pool (the next `--n-tests` layouts), exactly as in `train_on_pool.py`. Every trainer
created here has its replay memory wrapped in a `BiasedMemory` seeded with one optimal winning
episode per training layout: the LLE SAT solver finds the shortest joint plan in which every agent
reaches an exit, and that plan is replayed to produce an episode which is never evicted, so it
keeps being sampled during the whole training. The plans are cached in a JSON file stored next to
the layouts of the pool and keyed by layout file name, so that a pool is only ever solved once:
subsequent runs load that file, solve the layouts that are missing from it (if any), store it back
and carry on.

Hyperparameters are the best parameters of the 9x9 cooperative study of the algorithm, read from
`tuning/certified-cooperation.journal`, so that the biased runs use a configuration that was tuned *without* a bias: any improvement
is attributable to the demonstrations rather than to a re-tuning.

This script only produces the *biased* condition. The control condition is `train_on_pool.py` run
with the same `--pool-size`, `--n-tests`, `--n-steps` and seeds; its default `--logdir-prefix` is
empty, so the two sets of experiments never collide. Comparing them answers the two questions of
the study:

1. *Does the bias help?* Compare the train metrics of both conditions for a fixed algorithm and
   seed set.
2. *Does the bias transfer?* The test environment is the held-out pool, whose layouts are never
   solved, never trained on and never present in the bias. Compare the test `exit_rate` of both
   conditions. Per-layout evaluation on the *seen* layouts can be obtained afterwards with
   `scripts/test_policy_on_train_envs.py`.

Only value-based algorithms are in the scope of this script: policy gradient methods such as PPO
do not learn from a replay memory of past experience, so an optimal-demonstration bias is
meaningless for them.

Example:
    uv run python scripts/biased-pool-replay.py layouts/convergent-2 --n-tests 1000
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, cast, get_args

import dotenv
import optuna
import tuning
import typed_argparse as tap
from lle import World
from lle.solver import Solver
from marlenv import Episode, Transition
from marlenv.models.env import MARLEnv
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial
from tqdm import tqdm
from train_on_pool import PoolSpec, experiment_logdir, layout_files, make_env, parse_pool_spec
from tuning import make_trainer

import marl
from marl.algos import DQN
from marl.models import BiasedMemory, EpisodeMemory, ReplayMemory, TransitionMemory

logger = logging.getLogger(__name__)

SETTING = "cooperative"
STUDY_MAP_NAME = "9x9_agents3_lasers2"
"""The hyperparameters always come from the 9x9 cooperative variant of the tuning study."""
DEFAULT_STUDY_JOURNAL = Path("tunings/certified-cooperation.journal")
SOLUTIONS_FILE = "solutions.json"
"""Cached shortest plans, stored next to the layouts of the pool and keyed by layout file name."""
SOLUTIONS_FILE_NO_GEMS = "solutions-no-gems.json"
"""Plans that are not required to collect the gems are cached separately: they are shorter."""
Algo = Literal["vdn", "qmix", "dqn"]
"""Policy gradient methods have no replay memory to bias and are out of the scope of this script."""
ALGOS: tuple[Algo, ...] = get_args(Algo)


class Args(tap.TypedArgs):
    pool_dir: Path = tap.arg(positional=True, help="Directory containing the pool of maps to train on.")
    n_seeds: int = tap.arg("--n-seeds", default=10)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    n_jobs: int = tap.arg("--n-jobs", default=8)
    pool_size: int = tap.arg("--pool-size", default=500, help="Size of the training pool (positive integer).")
    n_tests: int = tap.arg("--n-tests", help="Number of held-out test maps (positive integer).")
    algos: list[Algo] = tap.arg("--algos", default=list(ALGOS), nargs="+", help="Value-based algorithms to train.")
    n_bias: int = tap.arg(
        "--n-bias",
        default=0,
        help="Number of training layouts to solve and bias towards. 0 means every training layout.",
    )
    bias_factor: float = tap.arg(
        "--bias-factor",
        default=1.0,
        help="Relative sampling weight of a biased item. 1.0 samples them like any other item.",
    )
    no_collect_gems: bool = tap.arg(
        "--no-collect-gems",
        default=False,
        help="Do not require the demonstrations to collect every gem before exiting.",
    )
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    study_journal: Path = tap.arg("--study-journal", default=DEFAULT_STUDY_JOURNAL)
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False)
    skip_existing: bool = tap.arg("--skip-existing", default=True)
    test_interval: int = tap.arg("--test-interval", default=50_000)
    logdir_prefix: str = tap.arg("--logdir-prefix", default="bias-", help="Prefix of the experiment log directories.")


def load_best_params(journal: Path, algo: Algo):
    """
    Read the best hyperparameters of an algorithm from the Optuna journal.

    The parameters are always those of the 9x9 cooperative variant of the study, whatever the pool
    that is trained on, so that every biased run of an algorithm starts from the same tuned
    configuration.
    """
    storage = JournalStorage(JournalFileBackend(journal.as_posix()))
    study_name = f"{algo.upper()}-{SETTING}-{STUDY_MAP_NAME}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError(f"Study {study_name!r} has no complete trials.")
    logger.info(f"Using best trial {study.best_trial.number} of {study_name} (value {study.best_value})")
    return study.best_params


def solutions_path(spec: PoolSpec, collect_gems: bool):
    """
    Path of the file caching the solutions of a pool, in the very directory of its layouts.

    @ai-generated
    """
    return spec.path / (SOLUTIONS_FILE if collect_gems else SOLUTIONS_FILE_NO_GEMS)


def load_solutions(path: Path) -> dict[str, list[list[int]] | None]:
    """
    Read the cached plans of a pool, mapping a layout file name to its joint actions per step.

    A `None` plan records a layout that the solver could not solve, so that a later run does not
    pay for solving it again. An absent or unreadable file simply yields an empty cache.

    @ai-generated
    """
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            solutions = json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse {path}; solving the layouts again.")
        return {}
    logger.info(f"Loaded {len(solutions)} cached solutions from {path}")
    return solutions


def save_solutions(path: Path, solutions: dict[str, list[list[int]] | None]):
    """
    Write the plans of a pool atomically, so that an interruption never leaves a truncated cache.

    @ai-generated
    """
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(solutions, f)
    tmp.replace(path)
    logger.info(f"Saved {len(solutions)} solutions to {path}")


def solve_layout(layout: Path, time_limit: int, collect_gems: bool):
    """
    Find a shortest winning joint plan of a layout with the LLE SAT solver.

    The plan is returned as the list of the joint actions to perform at each step, as integers.
    Returns `None` when the layout admits no plan within the time limit.

    @ai-generated
    """
    solver = Solver(World.from_file(layout.as_posix()), time_limit)
    plan = solver.find_shortest(collect_gems=collect_gems)
    if plan is None and collect_gems:
        plan = solver.find_shortest()
    if plan is None:
        return None
    return [[action.value for action in joint_action] for joint_action in plan]


def compute_solutions(spec: PoolSpec, layouts: list[Path], collect_gems: bool):
    """
    Return the plans of the given layouts, solving and caching the ones that are missing.

    The cache is shared by every run on that pool since it lives next to the layouts: only the
    layouts that have never been solved are given to the solver, and the file is rewritten as soon
    as new plans are found, including when the solving is interrupted.

    @ai-generated
    """
    path = solutions_path(spec, collect_gems)
    solutions = load_solutions(path)
    missing = [layout for layout in layouts if layout.name not in solutions]
    if len(missing) == 0:
        return solutions
    logger.info(f"Solving {len(missing)} layouts of {spec.path}")
    try:
        for layout in tqdm(missing, desc="Solving layouts", unit="layout"):
            solutions[layout.name] = solve_layout(layout, spec.time_limit, collect_gems)
    finally:
        save_solutions(path, solutions)
    return solutions


def replay_solution(spec: PoolSpec, index: int, plan: list[list[int]]):
    """
    Roll out a plan of the `index`th layout of the pool to produce an episode.

    The plan is replayed in a single-layout pool built with the very same configuration as the
    training environment, so that the resulting episode is indistinguishable from experience
    collected during training. Returns `None` when the replay is not a winning episode.

    @ai-generated
    """
    env = make_env(spec.path, 1, offset=index, time_limit=spec.time_limit).make()
    obs, state = env.reset()
    episode = Episode.new(obs, state)
    for actions in plan:
        step = env.step(actions)
        episode.add(Transition.from_step(obs, state, actions, step))
        obs, state = step.obs, step.state
    if not episode.is_finished or episode.metrics.get("exit_rate", 0.0) < 1.0:
        logger.warning(f"The plan of layout #{index} is not a winning episode; discarding it.")
        return None
    return episode


def make_demonstrations(spec: PoolSpec, n_layouts: int, collect_gems: bool):
    """
    Collect one winning episode for each of the first `n_layouts` layouts of the training pool.

    The plans come from the solution cache of the pool, which is completed and stored on disk
    before any episode is replayed. Unsolvable layouts are not admissible inputs: a pool that
    contains one is rejected outright rather than silently biasing the memory towards a subset of
    the training layouts.

    @ai-generated
    """
    layouts = layout_files(spec.path)
    if len(layouts) < n_layouts:
        raise ValueError(f"{spec.path} only contains {len(layouts)} layouts, cannot bias towards {n_layouts}.")
    layouts = layouts[:n_layouts]
    solutions = compute_solutions(spec, layouts, collect_gems)
    episodes = list[Episode]()
    for index, layout in enumerate(tqdm(layouts, desc="Replaying solutions", unit="layout")):
        plan = solutions[layout.name]
        episode = replay_solution(spec, index, plan) if plan is not None else None
        if episode is None:
            raise RuntimeError(f"Layout {index} ({layout.name}) of {spec.path} admits no winning plan!")
        episodes.append(episode)
    logger.info(f"Collected {len(episodes)} demonstrations.")
    return episodes


def bias_memory(
    base_memory: ReplayMemory[Transition] | ReplayMemory[Episode], demos: list[Episode], factor: float
) -> ReplayMemory:
    """
    Wrap a replay memory into one that permanently holds the demonstrations.

    Episode memories (recurrent networks) are biased with whole episodes, transition memories with
    the individual transitions of those same episodes. The wrapped memory keeps the capacity that
    the trainer asked for, so a biased run only differs from its control by the extra,
    never-evicted items.

    @ai-generated
    """
    match base_memory:
        case EpisodeMemory():
            return BiasedMemory(demos, base_memory, factor=factor)
        case TransitionMemory():
            return BiasedMemory([t for e in demos for t in e.transitions()], base_memory, factor=factor)
    raise TypeError(f"Cannot bias a memory of type {type(base_memory).__name__}.")


def get_experiment(args: Args, spec: PoolSpec, algo: Algo, demos: list[Episode]):
    """
    Create (or resume) and run the biased experiment of one algorithm.

    The bias lives in the trainer's `memory` attribute, which is *not* serialised with the
    experiment, so it is re-injected every time this script runs, including when resuming.
    """
    logdir = experiment_logdir(spec, algo, args.n_steps, args.pool_size, args.logdir_prefix)
    requested_seeds = range(args.start_seed, args.start_seed + args.n_seeds)
    if logdir.exists():
        exp = marl.Experiment[MARLEnv, DQN].load(logdir)
        completed_seeds = {run.seed for run in exp.runs if run.is_complete and run.seed in requested_seeds}
        seeds = [seed for seed in requested_seeds if seed not in completed_seeds]
        if args.dry_run:
            logger.info(f"[exists] {len(seeds)} runs of {spec.map_name} / {algo} -> {logdir}")
            return
        if len(seeds) == 0:
            if args.skip_existing:
                logger.info(f"Skipping complete experiment {logdir} ({len(completed_seeds)}/{args.n_seeds} runs)")
                return
            raise FileExistsError(f"Experiment directory already exists: {logdir}")
        logger.info(f"Resuming {logdir} with the missing seeds {seeds}")
    else:
        seeds = list(requested_seeds)
        if args.dry_run:
            logger.info(f"[new] {len(seeds)} runs of {spec.map_name} / {algo} -> {logdir}")
            return
        train_env = make_env(spec.path, args.pool_size, time_limit=spec.time_limit)
        test_env = make_env(spec.path, args.n_tests, offset=args.pool_size, time_limit=spec.time_limit)
        params = load_best_params(args.study_journal, algo)
        tuning_args = tuning.Args(pool_dirs=[spec.path], n_steps=args.n_steps)
        trainer = make_trainer(cast(optuna.Trial, FixedTrial(params)), algo, train_env, tuning_args)
        exp = marl.Experiment.create(train_env, trainer, test_env=test_env, logdir=logdir, n_steps=args.n_steps)
        logger.info(f"Created experiment in {exp.logdir} with parameters {params}")
    memory = bias_memory(exp.trainer.memory, demos, args.bias_factor)
    exp.trainer.memory = memory  # type:ignore
    exp.run(
        seeds=seeds,
        save_weights=True,
        save_actions=True,
        test_interval=args.test_interval,
        n_tests=args.n_tests,
        n_jobs=args.n_jobs,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=args.quiet,
        limit_torch_threads=False,
    )


def main(args: Args):
    if args.pool_size <= 0:
        raise ValueError(f"--pool-size must be a positive integer, got {args.pool_size}")
    if args.n_tests <= 0:
        raise ValueError(f"--n-tests must be a positive integer, got {args.n_tests}")
    if args.n_bias < 0 or args.n_bias > args.pool_size:
        raise ValueError(f"--n-bias must be in [0, {args.pool_size}], got {args.n_bias}")
    spec = parse_pool_spec(args.pool_dir)

    demos = []
    if not args.dry_run:
        demos = make_demonstrations(spec, args.n_bias or args.pool_size, not args.no_collect_gems)
    logger.info(f"Starting the biased-replay study on {spec.map_name}: {args.algos}")
    for algo in args.algos:
        get_experiment(args, spec, algo, demos)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("biased_pool_replay.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception(f"The biased-replay study failed with command line '{sys.argv}'.")
        raise
