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
from marlenv import DiscreteMARLEnv, Episode, Transition
from marlenv.models.env import MARLEnv
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial
from tqdm import tqdm
from train_on_pool import PoolSpec, experiment_logdir, layout_files, make_env, parse_pool_spec

import marl
from marl.algos import DQN
from marl.env import EnvConfig
from marl.models import BiasedMemory, EpisodeMemory, ReplayMemory, TransitionMemory

logger = logging.getLogger(__name__)

STUDY_LAYOUT_TYPE = "interdependent-2-9x9_agents3_lasers2"
"""The perspective-tuning layout whose best trials provide the replay-study hyperparameters."""
DEFAULT_STUDY_JOURNAL = Path("tunings", "perspective.journal")
SOLUTIONS_FILE = "solutions.json"
"""Cached shortest plans, stored next to the layouts of the pool and keyed by layout file name."""
SOLUTIONS_FILE_NO_GEMS = "solutions-no-gems.json"
"""Plans that are not required to collect the gems are cached separately: they are shorter."""
Algo = Literal["vdn", "qmix", "dqn"]
"""Policy gradient methods have no replay memory to bias and are out of the scope of this script."""
ALGOS: tuple[Algo, ...] = get_args(Algo)


class Args(tap.TypedArgs):
    pool_dirs: list[Path] = tap.arg(positional=True, help="Directory containing the pool of maps to train on.")
    n_seeds: int = tap.arg("--n-seeds", default=16)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    n_jobs: int = tap.arg("--n-jobs", default=1)
    pool_size: int = tap.arg("--pool-size", default=500, help="Size of the training pool (positive integer).")
    offset: int = tap.arg("--offset", default=1_000, help="Index of the first training layout (non-negative integer).")
    n_tests: int = tap.arg("--n-tests", default=500, help="Number of held-out test maps (positive integer).")
    algos: list[Algo] = tap.arg("--algos", nargs="+", help="Value-based algorithms to train.")
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
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    study_journal: Path = tap.arg("--study-journal", default=DEFAULT_STUDY_JOURNAL)
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False)
    test_interval: int = tap.arg("--test-interval", default=50_000)
    logdir_prefix: str = tap.arg("--logdir-prefix", default="bias-", help="Prefix of the experiment log directories.")

    @property
    def requested_seeds(self):
        return list(range(self.start_seed, self.start_seed + self.n_seeds))


def load_best_params(journal: Path, algo: Algo):
    """Read the best hyperparameters of an algorithm from the Optuna journal of `scripts/tuning.py`."""
    if not journal.exists():
        raise FileNotFoundError(f"No tuning journal at {journal}.")
    storage = JournalStorage(JournalFileBackend(journal.as_posix()))
    study_name = f"{algo.upper()}-{STUDY_LAYOUT_TYPE}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        available = sorted(summary.study_name for summary in optuna.get_all_study_summaries(storage))
        raise KeyError(f"{journal} has no study {study_name!r}. Available studies: {available}") from None
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError(f"Study {study_name!r} has no complete trials.")
    logger.info(
        f"Using best trial {study.best_trial.number} of {study_name} (value {study.best_value}, {len(complete_trials)} complete trials)"
    )
    return study.best_params


def make_journal_trainer(
    trial: optuna.Trial,
    algo: Algo,
    env: EnvConfig[DiscreteMARLEnv],
    demos: list[Episode],
    n_steps: int,
    factor: float,
):
    """
    Rebuild a tuned trainer from the journal parameters and bias its replay memory.

    The trainer is built by the very factory of the tuning study, so that the parameters read from
    the journal are replayed into the exact search space that produced them: the biased condition
    and its control therefore only differ by the demonstrations held in the memory, whose capacity
    remains the tuned `memory_size`.
    """
    if algo not in ("dqn", "vdn", "qmix"):
        raise NotImplementedError(f"Only value-based algorithms are in the scope of this study, got {algo!r}.")
    catch_all = {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": tuning.GAMMA}
    trainer = tuning.make_dqn_trainer(trial, algo, env, n_steps, catch_all)
    trainer.memory = bias_memory(trainer.memory, demos, factor)
    return trainer


def load_solutions(path: Path) -> dict[str, list[list[int]] | None]:
    """
    Read the cached plans of a pool, mapping a layout file name to its joint actions per step.

    A `None` plan records a layout that the solver could not solve, so that a later run does not
    pay for solving it again. An absent or unreadable file simply yields an empty cache.
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


def solve_layout(layout: Path, time_limit: int):
    """
    Find a shortest winning joint plan of a layout with the LLE SAT solver.

    The plan is returned as the list of the joint actions to perform at each step, as integers.
    Returns `None` when the layout admits no plan within the time limit.
    """
    solver = Solver(World.from_file(layout.as_posix()), time_limit)
    plan = solver.find_shortest(collect_gems=True)
    if plan is None:
        return None
    return [[action.value for action in joint_action] for joint_action in plan]


def compute_solutions(spec: PoolSpec, layouts: list[Path]):
    path = spec.path / SOLUTIONS_FILE
    solutions = load_solutions(path)
    missing = [layout for layout in layouts if layout.name not in solutions]
    if len(missing) == 0:
        return solutions
    logger.info(f"Solving {len(missing)} layouts of {spec.path}")
    try:
        for layout in tqdm(missing, desc="Solving layouts", unit="layout"):
            solutions[layout.name] = solve_layout(layout, spec.time_limit)
    finally:
        save_solutions(path, solutions)
    return solutions


def make_demonstrations(spec: PoolSpec, n_layouts: int, offset: int):
    """
    Collect one winning episode for the first `n_layouts` layouts of the training pool.

    The plans come from the solution cache of the pool, which is completed and stored on disk
    before any episode is replayed. Unsolvable layouts are not admissible inputs: a pool that
    contains one is rejected outright rather than silently biasing the memory towards a subset of
    the training layouts.

    @ai-edited
    """
    layouts = layout_files(spec.path)
    required_layouts = offset + n_layouts
    if len(layouts) < required_layouts:
        raise ValueError(f"{spec.path} only contains {len(layouts)} layouts, cannot select {n_layouts} from offset {offset}.")
    layouts = layouts[offset:required_layouts]
    solutions = compute_solutions(spec, layouts)
    episodes = list[Episode]()
    unsolved = [layout.name for layout in layouts if solutions[layout.name] is None]
    if len(unsolved) > 0:
        raise ValueError(f"No winning plan for {len(unsolved)} layouts in {spec.path}: {unsolved[:5]}")
    for index, layout in enumerate(tqdm(layouts, desc="Replaying solutions", unit="layout"), start=offset):
        plan = solutions[layout.name]
        assert plan is not None
        env = make_env(spec.path, 1, offset=index, time_limit=spec.time_limit).make()
        episodes.append(env.replay(plan))
    logger.info(f"Collected {len(episodes)} demonstrations.")
    return episodes


def bias_memory(base_memory: ReplayMemory[Transition] | ReplayMemory[Episode], demos: list[Episode], factor: float) -> ReplayMemory:
    """
    Wrap a replay memory into one that permanently holds the demonstrations.

    Episode memories (recurrent networks) are biased with whole episodes, transition memories with
    the individual transitions of those same episodes. Demonstrations are persisted once as a
    pickle artifact and materialized lazily in each worker. The wrapped memory keeps the capacity
    that the trainer asked for, so a biased run only differs from its control by the extra,
    never-evicted items.

    @ai-edited
    """
    if not isinstance(base_memory, (EpisodeMemory, TransitionMemory)):
        raise TypeError(f"Cannot bias a memory of type {type(base_memory).__name__}.")
    return BiasedMemory.from_episodes(demos, base_memory, factor=factor)


def get_experiment(args: Args, spec: PoolSpec, algo: Algo):
    """
    Create (or resume) the biased experiment of one algorithm.

    The bias lives in the trainer's `memory`, which is serialised with the experiment: an existing
    experiment is resumed as it is, demonstrations included, and the layouts of its pool are never
    solved again.
    """
    logdir = experiment_logdir(spec, algo, args.n_steps, args.pool_size, args.logdir_prefix)
    try:
        return marl.Experiment[MARLEnv, DQN].load(logdir)
    except FileNotFoundError:
        pass
    params = load_best_params(args.study_journal, algo)
    demos = make_demonstrations(spec, args.n_bias or args.pool_size, args.offset)
    train_env = make_env(spec.path, args.pool_size, offset=args.offset, time_limit=spec.time_limit)
    test_env = make_env(spec.path, args.n_tests, offset=args.offset + args.pool_size, time_limit=spec.time_limit)
    trial = cast(optuna.Trial, FixedTrial(params))
    trainer = make_journal_trainer(trial, algo, train_env, demos, args.n_steps, args.bias_factor)
    exp = marl.Experiment.create(train_env, trainer, test_env=test_env, logdir=logdir, n_steps=args.n_steps)
    logger.info(f"Created experiment in {exp.logdir} with parameters {params}")
    return exp


def run_experiment(exp: marl.Experiment, args: Args):
    completed_seeds = {run.seed for run in exp.runs if run.is_complete and run.seed in args.requested_seeds}
    seeds = list(set(args.requested_seeds) - set(completed_seeds))
    seeds.sort()
    if len(seeds) == 0:
        logger.info(f"All requested seeds are complete in {exp.logdir}")
        return
    logger.info(f"Resuming {exp.logdir} with the missing seeds {seeds}")
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
        limit_torch_threads=None,
    )


def main(args: Args):
    """Validate the selected pools and launch or preview the requested runs. @ai-edited"""
    if args.n_seeds <= 0:
        raise ValueError(f"--n-seeds must be positive, got {args.n_seeds}")
    if args.n_jobs <= 0:
        raise ValueError(f"--n-jobs must be positive, got {args.n_jobs}")
    if args.bias_factor <= 0:
        raise ValueError(f"--bias-factor must be positive, got {args.bias_factor}")
    if args.offset < 0:
        raise ValueError(f"--offset must be a non-negative integer, got {args.offset}")
    if args.pool_size <= 0:
        raise ValueError(f"--pool-size must be a positive integer, got {args.pool_size}")
    if args.n_tests <= 0:
        raise ValueError(f"--n-tests must be a positive integer, got {args.n_tests}")
    if args.n_bias < 0 or args.n_bias > args.pool_size:
        raise ValueError(f"--n-bias must be in [0, {args.pool_size}], got {args.n_bias}")
    if not args.pool_dirs:
        raise ValueError("Provide at least one pool directory (e.g. layouts/canonicals/asymmetric).")
    for pool_dir in args.pool_dirs:
        if not pool_dir.is_dir():
            raise ValueError(f"Layout pool is not a directory: {pool_dir}")
        layouts = layout_files(pool_dir)
        required = args.offset + args.pool_size + args.n_tests
        if len(layouts) < required:
            raise ValueError(f"{pool_dir} has {len(layouts)} layouts; need {required} for training and held-out tests.")
        spec = parse_pool_spec(pool_dir)
        logger.info(f"Starting the biased-replay study on {spec.map_name}: {args.algos}")
        for algo in args.algos:
            logdir = experiment_logdir(spec, algo, args.n_steps, args.pool_size, args.logdir_prefix)
            if args.dry_run:
                if logdir.exists():
                    exp = marl.Experiment[MARLEnv, DQN].load(logdir)
                    missing = set(args.requested_seeds) - {run.seed for run in exp.runs if run.is_complete}
                    logger.info(f"[exists] {len(missing)} missing runs -> {logdir}")
                else:
                    load_best_params(args.study_journal, algo)
                    logger.info(f"[new] {args.n_seeds} runs -> {logdir}")
                continue
            exp = get_experiment(args, spec, algo)
            run_experiment(exp, args)


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
