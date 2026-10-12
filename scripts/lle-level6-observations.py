import logging
import os
import sys
from pathlib import Path
from typing import Literal, cast

import dotenv
import optuna
import tuning
import typed_argparse as tap
from lle import World
from lle.solver import Solver
from marlenv import Episode
from marlenv.models.env import MARLEnv
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial

import marl
from marl.algos import DQN, VDN, QMix
from marl.env import LLEConfig
from marl.models import BiasedMemory, TransitionMemory
from marl.nn import mixers, model_bank
from marl.runners import parallel_run
from marl.utils.tuning import suggest

logger = logging.getLogger(__name__)

PERSPECTIVE_STUDY_LAYOUT_TYPE = "interdependent-2-9x9_agents3_lasers2"
"""The perspective-tuning layout whose best trials provide the hyperparameters of the perspective runs."""
DEFAULT_PERSPECTIVE_JOURNAL = Path("tunings", "perspective.journal")
LAYERED_STUDY_LAYOUT_TYPE = "cooperative-9x9_agents3_lasers2"
"""The certified-cooperation layout whose best trials provide the hyperparameters of the layered runs."""
DEFAULT_LAYERED_JOURNAL = Path("tunings", "certified-cooperation.journal")
CERTIFIED_MEMORY_SIZE = 50_000
"""The certified-cooperation study did not tune the memory: DQN then defaulted to a 50k transition memory."""
LEVEL = 6
Algo = Literal["vdn", "qmix", "dqn"]
"""Only value-based algorithms have a replay memory to bias."""
ObsType = Literal["layered", "perspective"]
ALGOS: list[Algo] = ["vdn", "qmix", "dqn"]
OBS_TYPES: list[ObsType] = ["layered", "perspective"]


class Args(tap.TypedArgs):
    n_seeds: int = tap.arg("--n-seeds", default=16)
    start_seed: int = tap.arg("--start-seed", default=0)
    n_steps: int = tap.arg("--n-steps", default=1_000_000)
    n_jobs: int = tap.arg("--n-jobs", default=1)
    n_tests: int = tap.arg("--n-tests", default=10, help="Number of test episodes at each test interval.")
    test_interval: int = tap.arg("--test-interval", default=10_000)
    algos: list[Algo] = tap.arg("--algos", nargs="+", default=ALGOS)
    obs_types: list[ObsType] = tap.arg("--obs-types", nargs="+", default=OBS_TYPES)
    n_bias: int = tap.arg(
        "--n-bias",
        default=1,
        help="Number of copies of the solved level-6 episode in the replay memory. 0 uses no demonstration.",
    )
    bias_factor: float = tap.arg(
        "--bias-factor",
        default=1.0,
        help="Relative sampling weight of a biased item. 1.0 samples them like any other item.",
    )
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    layered_journal: Path = tap.arg("--layered-journal", default=DEFAULT_LAYERED_JOURNAL)
    perspective_journal: Path = tap.arg("--perspective-journal", default=DEFAULT_PERSPECTIVE_JOURNAL)
    quiet: bool = tap.arg("--quiet", default=True)
    dry_run: bool = tap.arg("--dry-run", default=False)

    @property
    def requested_seeds(self):
        return list(range(self.start_seed, self.start_seed + self.n_seeds))

    def study(self, obs_type: ObsType) -> tuple[Path, str]:
        """The journal and layout type of the study that tuned the given observation type. @ai-generated"""
        if obs_type == "layered":
            return self.layered_journal, LAYERED_STUDY_LAYOUT_TYPE
        return self.perspective_journal, PERSPECTIVE_STUDY_LAYOUT_TYPE


def load_best_params(journal: Path, layout_type: str, algo: Algo):
    """Read the best hyperparameters of an algorithm from an Optuna journal. @ai-generated"""
    if not journal.exists():
        raise FileNotFoundError(f"No tuning journal at {journal}.")
    storage = JournalStorage(JournalFileBackend(journal.as_posix()))
    study_name = f"{algo.upper()}-{layout_type}"
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


def experiment_logdir(obs_type: ObsType, algo: Algo, args: Args):
    """
    Name the experiment after its condition. The bias count and factor only appear when they differ
    from their defaults, so that distinct bias settings never share a directory.

    @ai-generated
    """
    millions = args.n_steps / 1_000_000
    steps = f"{int(millions)}M" if millions.is_integer() else f"{millions:.1f}M"
    if args.n_bias == 0:
        prefix = "unbiased-"
    else:
        prefix = "bias-"
        if args.n_bias != 1:
            prefix += f"n{args.n_bias}-"
        if args.bias_factor != 1.0:
            prefix += f"f{args.bias_factor:g}-"
    return Path("logs", f"{prefix}lle-level{LEVEL}-{obs_type}-{algo}-{steps}")


def make_demonstration(env: LLEConfig) -> Episode:
    """
    Solve level 6 with the LLE SAT solver and replay the shortest plan that collects every gem.

    @ai-generated
    """
    assert env.time_limit is not None
    plan = Solver(World.level(LEVEL), env.time_limit).find_shortest(collect_gems=True)
    if plan is None:
        raise RuntimeError(f"The solver found no plan for level {LEVEL} within {env.time_limit} steps.")
    episode = env.make().replay([[action.value for action in joint_action] for joint_action in plan])
    logger.info(f"Solved level {LEVEL} in {len(episode)} steps (score {episode.score}).")
    return episode


def hidden_sizes(trial: optuna.Trial, prefix: str) -> list[int]:
    """Uniform-width MLP architecture of the certified-cooperation search space. @ai-generated"""
    return [trial.suggest_int(f"{prefix}.size", 32, 512, step=32)] * trial.suggest_int(f"{prefix}.n_layers", 1, 5)


def make_certified_trainer(trial: optuna.Trial, algo: Algo, env: LLEConfig, catch_all: dict):
    """
    Rebuild a value-based trainer in the search space of the certified-cooperation study.

    This replicates `make_dqn_trainer` of `scripts/tuning.py` at commit d2ea0990, which produced
    `tunings/certified-cooperation.journal`: the network width and depth, the training policy, and the
    target updater are searched, while the memory keeps the then-default capacity.

    @ai-generated
    """
    qnetwork = model_bank.qnetworks.from_env(
        env,
        independent=True,
        duelling=trial.suggest_categorical("qnetwork.duelling", [True, False]),
        noisy=trial.suggest_categorical("qnetwork.noisy", [False, True]),
        mlp_sizes=hidden_sizes(trial, "qnetwork"),
    )
    kwargs = {
        "qnetwork": qnetwork,
        "memory": TransitionMemory(CERTIFIED_MEMORY_SIZE),
        "test_policy": marl.policy.ArgMax(),
        "vbe": None,
        "catch_all": catch_all,
    }
    match algo:
        case "dqn":
            return suggest(DQN, trial, mixer=None, **kwargs)
        case "vdn":
            return suggest(VDN, trial, **kwargs)
        case "qmix":
            return suggest(QMix, trial, mixer=mixers.QMix.from_env(env), **kwargs)
    raise NotImplementedError(f"Unsupported algorithm {algo!r}.")


def make_trainer(params: dict, algo: Algo, env: LLEConfig, n_steps: int, demos: list[Episode], factor: float):
    """
    Rebuild a tuned trainer by replaying the journal parameters into the search space that produced them,
    then bias its replay memory with the demonstrations, if any.

    Layered runs use the certified-cooperation search space, perspective runs the current one of
    `scripts/tuning.py`. The online memory retains its tuned capacity; the demonstrations are
    additional, never-evicted storage.

    @ai-generated
    """
    trial = cast(optuna.Trial, FixedTrial(params))
    catch_all = {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": tuning.GAMMA}
    if env.obs_type == "layered":
        trainer = make_certified_trainer(trial, algo, env, catch_all)
    else:
        trainer = tuning.make_dqn_trainer(trial, algo, env, n_steps, catch_all)
    if len(demos) > 0:
        assert isinstance(trainer.memory, TransitionMemory)
        trainer.memory = BiasedMemory.from_episodes(demos, trainer.memory, factor=factor)
    return trainer


def get_experiment(args: Args, obs_type: ObsType, algo: Algo):
    """
    Load the experiment of an (observation, algorithm) pair, or create it if it does not exist.

    The bias lives in the trainer's memory, which is serialized with the experiment.

    @ai-generated
    """
    logdir = experiment_logdir(obs_type, algo, args)
    try:
        return marl.Experiment[MARLEnv, DQN].load(logdir)
    except FileNotFoundError:
        pass
    params = load_best_params(*args.study(obs_type), algo)
    env = LLEConfig(LEVEL, obs_type=obs_type, state_type="flattened")
    demos = [make_demonstration(env)] * args.n_bias if args.n_bias > 0 else []
    trainer = make_trainer(params, algo, env, args.n_steps, demos, args.bias_factor)
    exp = marl.Experiment.create(env, trainer, logdir=logdir, n_steps=args.n_steps)
    logger.info(f"Created experiment in {exp.logdir} with parameters {params}")
    return exp


def missing_runs(exp: marl.Experiment, args: Args) -> list[marl.Run]:
    """
    Create the runs of the requested seeds that are not complete in the experiment.

    Incomplete seeds restart from step zero: `create_runs` rewrites their `run.json`.

    @ai-generated
    """
    completed_seeds = {run.seed for run in exp.runs if run.is_complete}
    seeds = sorted(set(args.requested_seeds) - completed_seeds)
    if len(seeds) == 0:
        logger.info(f"All requested seeds are complete in {exp.logdir}")
        return []
    logger.info(f"Queueing {exp.logdir} with the missing seeds {seeds}")
    return exp.create_runs(seeds, n_tests=args.n_tests, test_interval=args.test_interval, save_weights=True, save_actions=True)


def main(args: Args):
    """
    Train every (observation type, algorithm) pair on LLE level 6, sharing a single run queue.

    @ai-generated
    """
    if args.n_seeds <= 0:
        raise ValueError(f"--n-seeds must be positive, got {args.n_seeds}")
    if args.n_jobs <= 0:
        raise ValueError(f"--n-jobs must be positive, got {args.n_jobs}")
    if args.n_bias < 0:
        raise ValueError(f"--n-bias must be non-negative, got {args.n_bias}")
    if args.bias_factor <= 0:
        raise ValueError(f"--bias-factor must be positive, got {args.bias_factor}")
    runs = list[marl.Run]()
    for obs_type in args.obs_types:
        for algo in args.algos:
            logdir = experiment_logdir(obs_type, algo, args)
            if args.dry_run:
                if logdir.exists():
                    exp = marl.Experiment.load(logdir)
                    missing = set(args.requested_seeds) - {run.seed for run in exp.runs if run.is_complete}
                    logger.info(f"[exists] {len(missing)} missing runs -> {logdir}")
                else:
                    load_best_params(*args.study(obs_type), algo)
                    logger.info(f"[new] {args.n_seeds} runs -> {logdir}")
                continue
            exp = get_experiment(args, obs_type, algo)
            runs += missing_runs(exp, args)
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
        handlers=[logging.FileHandler("lle_level6_observations.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception(f"The LLE level 6 observation study failed with command line '{sys.argv}'.")
        raise
