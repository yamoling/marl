"""
Train one experiment per entry of `LEVELS`, sequentially.

Each entry declares a level (an LLE built-in level or a map file) together with its
environment, algorithm and training parameters. Experiments are run one after the
other: `Experiment.run` blocks until every seed of a level has terminated, so the next
level only starts once the current one is entirely finished.

Within a level, all `seeds` train in parallel across GPUs (`n_jobs="all"` by default,
i.e. one worker per seed), according to `gpu_strategy` and `disabled_gpus`. Note that
`parallel_run` submits the first seed *alone* to measure its GPU memory usage, and only
submits the remaining seeds once that measurement stabilises: the first minute or so of
every level is single-run by design.

Edit the CONFIG block below to change what is trained. The resources and run lengths
can also be overridden on the command line without touching the file, e.g.

    python scripts/train_levels.py --dry-run
    python scripts/train_levels.py --levels lift3 --seeds 1 --n-jobs 1 --n-steps 2000 --tag smoke
    python scripts/train_levels.py --n-jobs 4 --disabled-gpus 0 1 2 3
"""

import logging
import os
import sys
import time
from collections.abc import Collection, Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Literal

import dotenv
import typed_argparse as tap
from lle.observations import ObservationTypeLiteral

import marl
from marl import algos
from marl.env import LLEConfig
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import EpsilonGreedy
from marl.utils import schedule

Algo = Literal["vdn", "qmix", "dqn"]


@dataclass(frozen=True)
class Exploration:
    """Epsilon-greedy exploration schedule.

    `DQN` defaults to a constant epsilon of 0.1, which never anneals: the agent keeps
    taking 10% random actions forever, and starts out too greedy to find a long
    button-then-lift sequence. A decaying schedule explores early and exploits later.
    """

    kind: Literal["linear", "exp", "constant"] = "linear"
    start: float = 1.0
    end: float = 0.05
    decay: float = 0.1
    """Annealing horizon, in environment steps. If <= 1, a fraction of the level's `n_steps`."""


@dataclass(frozen=True)
class LevelSpec:
    """A single (level, parameters) combination to train."""

    level: Literal[1, 2, 3, 4, 5, 6] | str
    """An LLE built-in level (1..6) or a path to a map file."""
    _: KW_ONLY
    label: str | None = None
    """Name used in the logdir. Derived from `level` when None."""
    # Environment (-> LLEConfig)
    time_limit: int | None = 32
    obs_type: ObservationTypeLiteral = "flattened"
    state_type: ObservationTypeLiteral = "flattened"
    pbrs: bool = False
    # Algorithm
    algo: Algo = "vdn"
    recurrent: bool = False
    independent: bool = False
    exploration: Exploration | None = None
    """Epsilon schedule. Falls back to `EXPLORATION`. Ignored if `trainer_params["train_policy"]` is set."""
    trainer_params: Mapping[str, Any] = field(default_factory=dict)
    """Extra trainer kwargs: lr, batch_size, memory_size, train_policy, gamma, train_interval, ..."""
    # Run parameters. When None, the value from `DEFAULTS` is used.
    n_steps: int | None = None
    seeds: int | Collection[int] | None = None
    test_interval: int | None = None
    n_tests: int | None = None
    n_jobs: int | None = None
    """Cap on the seeds trained simultaneously. None trains them all at once (see `Resources.n_jobs`)."""


@dataclass
class Resources:
    """Defaults shared by every level, overridable on the command line."""

    seeds: int | Collection[int] = 10
    n_jobs: int | Literal["all"] = 8
    """Maximum seeds trained simultaneously within a level, capped by the seed count.

    Set to "all" to always run one worker per seed. The default of 6 is a ceiling chosen to fit
    the enabled GPUs: `scatter_plan` raises when they cannot hold `n_jobs` runs at once.

    Levels stay sequential regardless: this only controls the concurrency *inside* a level. When
    a level has more seeds than `n_jobs`, the pool is rolling rather than batched -- the next seed
    starts as soon as any running one finishes, so `n_jobs` runs are in flight throughout.
    """
    gpu_strategy: Literal["scatter", "group"] = "scatter"
    disabled_gpus: list[int] = field(default_factory=list)
    n_steps: int = 1_000_000
    test_interval: int = 10_000
    n_tests: int = 10
    save_weights: bool = True
    save_actions: bool = False
    quiet: bool = True


# ================================== CONFIG ==================================
LEVELS: list[LevelSpec] = [
    LevelSpec(level=3, label="lvl3", time_limit=64),
    LevelSpec("maps/lvl3-liftversion.toml", label="lvl3-lift", time_limit=128),
    # LevelSpec("maps/lvl3-lift.toml", label="lvl3-lift", time_limit=64),
    # LevelSpec("lift3.toml", label="lift3", algo="qmix"),
    # LevelSpec("lift3.toml", label="lift3-lr5e4", trainer_params={"lr": 5e-4, "batch_size": 128}),
    # LevelSpec("lift3.toml", label="lift3-slow-eps", exploration=Exploration(end=0.02, decay=0.3)),
    # LevelSpec("lift3.toml", label="lift3-exp-eps", exploration=Exploration(kind="exp", decay=200_000)),
    # LevelSpec("lift3.toml", label="lift3-const-eps", exploration=Exploration(kind="constant", start=0.1)),
    # LevelSpec(6, label="lvl6", obs_type="layered", time_limit=None, n_steps=2_000_000),
]

SELECTED: list[str] | None = None
"""Labels to run. None runs every entry of `LEVELS`. Overridden by `--levels`."""

EXPLORATION = Exploration(kind="linear", start=1.0, end=0.05, decay=0.5)
"""Default epsilon schedule: anneal 1.0 -> 0.05 over the first half of each level's steps.

Tuned on the laser maps, Optuna picked anneal horizons of 124k, 415k and 701k steps for a
400k-step budget (31%, 104% and 175% of training): on sparse-reward LLE tasks it consistently
preferred long exploration. Lift/button is sparser still, so anything below ~25% risks
freezing the policy before it has ever seen a button-then-lift reward.
"""

DEFAULTS = Resources()
# ============================================================================


class Args(tap.TypedArgs):
    levels: list[str] = tap.arg("--levels", default=[], nargs="*", help="Only run these labels (overrides SELECTED).")
    seeds: int | None = tap.arg("--seeds", default=None, help="Number of seeds per level.")
    n_jobs: int | None = tap.arg(
        "--n-jobs",
        default=None,
        help="Seeds trained in parallel within a level. Defaults to all of them. Levels stay sequential.",
    )
    gpu_strategy: Literal["scatter", "group"] | None = tap.arg("--gpu-strategy", default=None)
    disabled_gpus: list[int] | None = tap.arg("--disabled-gpus", default=None, nargs="*")
    n_steps: int | None = tap.arg("--n-steps", default=None)
    test_interval: int | None = tap.arg("--test-interval", default=None)
    n_tests: int | None = tap.arg("--n-tests", default=None)
    tag: str = tap.arg("--tag", default="", help="Suffix appended to every logdir, to keep runs apart.")
    # `typed_argparse` only produces store_true flags, so each boolean override needs
    # both directions to be expressible. When neither is given, `DEFAULTS` applies.
    save_actions: bool = tap.arg("--save-actions", default=False, help="Save the test actions.")
    no_save_actions: bool = tap.arg("--no-save-actions", default=False)
    quiet: bool = tap.arg("--quiet", default=False, help="Disable progress bars and console logs.")
    no_quiet: bool = tap.arg("--no-quiet", default=False)
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only show what would be trained.")
    no_skip_existing: bool = tap.arg(
        "--no-skip-existing",
        default=False,
        help="Fail instead of skipping when a level is already fully trained.",
    )


def spec_label(spec: LevelSpec):
    if spec.label is not None:
        return spec.label
    match spec.level:
        case int(level):
            return f"lvl{level}"
        case str(path):
            return Path(path).stem


def logdir_for(spec: LevelSpec, tag: str):
    name = "-".join(part for part in (spec_label(spec), spec.algo, tag) if part)
    return Path("logs") / name


def build_env(spec: LevelSpec):
    return LLEConfig(
        spec.level,
        obs_type=spec.obs_type,
        state_type=spec.state_type,
        pbrs=spec.pbrs,
        time_limit=spec.time_limit,
    )


def make_exploration_policy(exploration: Exploration, n_steps: int):
    """Build the epsilon-greedy training policy. The schedule is annealed against the
    absolute environment time step (`DQN._update` -> `Policy.update(time_step)`)."""
    if exploration.kind == "constant":
        return EpsilonGreedy.constant(exploration.start)
    decay = exploration.decay
    decay_steps = max(2, int(decay * n_steps if decay <= 1.0 else decay))
    match exploration.kind:
        case "linear":
            return EpsilonGreedy.linear(exploration.start, exploration.end, decay_steps)
        case "exp":
            # ExpSchedule computes `start * (end / start) ** (t / (n - 1))`: both bounds must be > 0.
            if exploration.start <= 0.0 or exploration.end <= 0.0:
                raise ValueError(f"Exponential epsilon requires start and end > 0, got {exploration}.")
            return EpsilonGreedy.exponential(exploration.start, exploration.end, decay_steps)
        case other:
            raise ValueError(f"Unknown exploration kind {other!r}. Expected 'linear', 'exp' or 'constant'.")


def describe_policy(train_policy: Any) -> str:
    if not isinstance(train_policy, EpsilonGreedy):
        return f"policy {train_policy.name}"
    eps = train_policy.epsilon
    if isinstance(eps, schedule.ConstantSchedule):
        return f"epsilon constant {eps.start_value}"
    return f"epsilon {eps.name} {eps.start_value} -> {eps.end_value} over {eps.n_steps} steps"


def build_trainer(spec: LevelSpec, env: LLEConfig, n_steps: int):
    qnetwork = qnetworks.from_env(env, recurrent=spec.recurrent, independent=spec.independent)
    params = dict(spec.trainer_params)
    # An explicit `train_policy` in `trainer_params` always wins.
    if "train_policy" not in params:
        exploration = spec.exploration if spec.exploration is not None else EXPLORATION
        params["train_policy"] = make_exploration_policy(exploration, n_steps)
    match spec.algo:
        case "vdn":
            # `VDN.mixer` is `init=False`: it must not be passed here.
            return algos.VDN(qnetwork, **params)
        case "qmix":
            params.setdefault("mixer", mixers.QMix.from_env(env))
            return algos.QMix(qnetwork, **params)
        case "dqn":
            return algos.DQN(qnetwork, **params)
        case other:
            raise ValueError(f"Unknown algo {other!r}. Expected one of 'vdn', 'qmix', 'dqn'.")


def resolve[T](*candidates: T | None) -> T:
    """First non-None candidate, in order of decreasing priority (CLI, then spec, then defaults)."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise ValueError("No value provided")


def resolve_flag(enable: bool, disable: bool, default: bool) -> bool:
    """Resolve a boolean expressed as a pair of `--flag` / `--no-flag` store_true options."""
    if enable and disable:
        raise ValueError("Conflicting flags: cannot both enable and disable the same option.")
    if enable:
        return True
    if disable:
        return False
    return default


def as_seed_list(seeds: int | Collection[int]) -> list[int]:
    if isinstance(seeds, int):
        return list(range(seeds))
    return list(seeds)


def effective_n_jobs(spec: LevelSpec, args: Args, seeds: Collection[int]) -> int:
    """Seeds of this level to train simultaneously. "all" means one worker per seed.

    Never exceeds the number of seeds actually being trained: a resume with a single missing
    seed uses a single worker rather than leaving `n_jobs - 1` of them idle.
    """
    n_jobs = resolve(args.n_jobs, spec.n_jobs, DEFAULTS.n_jobs)
    if n_jobs == "all":
        return max(1, len(seeds))
    return max(1, min(n_jobs, len(seeds)))


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_spec(spec: LevelSpec, args: Args, index: int = 0, total: int = 0):
    """Train a single level. Returns "done", "partial", "skipped" or "dry-run"."""
    label = spec_label(spec)
    logdir = logdir_for(spec, args.tag)
    n_steps = resolve(args.n_steps, spec.n_steps, DEFAULTS.n_steps)
    seeds = as_seed_list(resolve(args.seeds, spec.seeds, DEFAULTS.seeds))
    test_interval = resolve(args.test_interval, spec.test_interval, DEFAULTS.test_interval)
    n_tests = resolve(args.n_tests, spec.n_tests, DEFAULTS.n_tests)
    prefix = f"Level {index}/{total} {label!r}" if total else f"Level {label!r}"

    if logdir.exists():
        exp = marl.Experiment.load(logdir)
        completed_seeds = {run.seed for run in exp.runs if run.is_complete}
        seeds = [seed for seed in seeds if seed not in completed_seeds]
        if args.dry_run:
            n_jobs = effective_n_jobs(spec, args, seeds)
            logging.info(f"[exists] {label} / {spec.algo} / {len(seeds)} missing seeds / n_jobs {n_jobs} -> {logdir}")
            return "dry-run"
        if len(seeds) == 0:
            if args.no_skip_existing:
                raise FileExistsError(f"Experiment directory already exists: {logdir}")
            logging.info(f"Skipping existing experiment {logdir} ({len(completed_seeds)} runs complete).")
            return "skipped"
        logging.info(f"Resuming {logdir}: {len(completed_seeds)} runs complete, starting missing seeds {seeds}.")
    else:
        # Building the env and the trainer validates the map path and the observation shape.
        env = build_env(spec)
        trainer = build_trainer(spec, env, n_steps)
        schedule = describe_policy(trainer.train_policy)
        if args.dry_run:
            n_jobs = effective_n_jobs(spec, args, seeds)
            logging.info(
                f"[new] {label} / {spec.algo} / {len(seeds)} seeds / n_jobs {n_jobs} / {n_steps} steps"
                f" / {schedule} -> {logdir}"
            )
            return "dry-run"
        exp = marl.Experiment.create(env, trainer, logdir=logdir, n_steps=n_steps)
        logging.info(f"Created experiment {exp.logdir} ({trainer.name} on {env.name}, {schedule}).")

    n_jobs = effective_n_jobs(spec, args, seeds)
    # `Experiment.run` falls back to `sequential_run` when either of these is <= 1.
    if n_jobs <= 1 or len(seeds) <= 1:
        logging.warning(
            f"{prefix}: training {len(seeds)} seed(s) with n_jobs={n_jobs}, i.e. sequentially, not in parallel."
        )
    else:
        logging.info(
            f"{prefix} ({spec.algo}): training {len(seeds)} seeds with n_jobs={n_jobs} (parallel),"
            f" {n_steps} steps -> {logdir}"
        )

    start_time = time.time()
    exp.run(
        seeds=seeds,
        gpu_strategy=resolve(args.gpu_strategy, DEFAULTS.gpu_strategy),
        save_weights=DEFAULTS.save_weights,
        save_actions=resolve_flag(args.save_actions, args.no_save_actions, DEFAULTS.save_actions),
        n_tests=n_tests,
        test_interval=test_interval,
        quiet=resolve_flag(args.quiet, args.no_quiet, DEFAULTS.quiet),
        n_jobs=n_jobs,
        disabled_gpus=resolve(args.disabled_gpus, DEFAULTS.disabled_gpus),
    )
    duration = format_duration(time.time() - start_time)

    # `parallel_run` logs and swallows the exceptions of individual runs, and also the
    # `RuntimeError` raised by `scatter_plan` when the enabled GPUs cannot fit `n_jobs` runs.
    # `exp.run` therefore returns normally even when nothing was trained: check the runs instead.
    completed_seeds = {run.seed for run in marl.Experiment.load(logdir).runs if run.is_complete}
    missing = [seed for seed in seeds if seed not in completed_seeds]
    # The next level only starts once this one has fully terminated: `exp.run` blocks above.
    next_level = " Starting next level." if index < total else ""
    if missing:
        logging.error(
            f"{prefix} finished in {duration} — {len(seeds) - len(missing)}/{len(seeds)} runs complete,"
            f" missing seeds {missing}.{next_level}"
        )
        return "partial"
    logging.info(f"{prefix} finished in {duration} — {len(seeds)}/{len(seeds)} runs complete.{next_level}")
    return "done"


def select_specs(args: Args):
    wanted = args.levels or SELECTED
    if not wanted:
        return list(LEVELS)
    specs = [spec for spec in LEVELS if spec_label(spec) in wanted or logdir_for(spec, "").name in wanted]
    unknown = set(wanted) - {spec_label(spec) for spec in specs} - {logdir_for(spec, "").name for spec in specs}
    if unknown:
        raise ValueError(f"Unknown level labels: {sorted(unknown)}. Available: {[spec_label(s) for s in LEVELS]}")
    return specs


def main(args: Args):
    specs = select_specs(args)
    logging.info(f"Training {len(specs)} levels sequentially: {[spec_label(spec) for spec in specs]}")

    outcomes = dict[str, str]()
    for i, spec in enumerate(specs, start=1):
        label = logdir_for(spec, args.tag).name
        try:
            outcomes[label] = run_spec(spec, args, index=i, total=len(specs))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Keep going: a broken level should not cancel the remaining ones.
            logging.error(f"Level {label} failed: {e}", exc_info=True)
            outcomes[label] = "failed"

    logging.info("Summary:")
    for label, outcome in outcomes.items():
        logging.info(f"  {outcome:<8} {label}")
    if not {"failed", "partial"}.isdisjoint(outcomes.values()):
        sys.exit(1)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("train_levels.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.error(
            f"An error occurred while running the levels with command line '{sys.argv}'.\nError: {e}", exc_info=True
        )
        raise
