import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dotenv
import optuna
import typed_argparse as tap
from lle import World
from marlenv import DiscreteMARLEnv
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import marl
from marl.algos import DQN, PPO, VDN, HardUpdate, QMix, SoftUpdate, TargetParametersUpdater
from marl.env import EnvConfig, LLEPool
from marl.models import Policy
from marl.nn import mixers, model_bank
from marl.utils import Schedule
from marl.utils.tuning import suggest

Algo = Literal["vdn", "qmix", "dqn", "mappo", "ippo", "qplex"]
Setting = Literal["cooperative", "independent"]

ALGOS: tuple[Algo, ...] = ("vdn", "qmix", "dqn")
DEFAULT_POOL_DIR = Path("layouts", "4-agents-inter-2")
TRAIN_POOL_SIZE = 2_500
TEST_POOL_SIZE = 2_500
"""The layouts in [TRAIN_POOL_SIZE + TEST_POOL_SIZE, ...[ are reserved for the actual experiments."""
GAMMA = 0.99
STORAGE_FILE = "tuning.journal"
MAX_CONSECUTIVE_FAILURES = 3
"""Number of consecutive scheduling rounds without a single completed trial before giving up."""

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolSpec:
    path: Path
    time_limit: int
    layout_type: str


class Args(tap.TypedArgs):
    pool_dirs: list[Path] = tap.arg(
        positional=True,
        nargs="*",
        default=[DEFAULT_POOL_DIR],
        help=(
            f"One or more directories containing at least {TRAIN_POOL_SIZE + TEST_POOL_SIZE:,} layouts "
            f"({TRAIN_POOL_SIZE:,} training and {TEST_POOL_SIZE:,} evaluation)."
        ),
    )
    n_jobs: int = tap.arg(
        "--n-jobs",
        default=1,
        help="Number of parallel Optuna jobs to run.",
    )
    n_steps: int = tap.arg(
        "--n-steps",
        default=1_000_000,
        help="Number of training steps per trial.",
    )
    disabled_gpus: list[int] = tap.arg(
        "--disabled-gpus",
        nargs="*",
        default=[],
        help="GPUs to disable.",
    )
    gpu_strategy: Literal["scatter", "group"] = tap.arg(
        "--gpu-strategy",
        default="scatter",
        help="GPU strategy to use.",
    )
    algos: list[Algo] = tap.arg(
        "--algos",
        nargs="*",
        default=list(ALGOS),
        help="Algorithms to tune.",
    )
    n_seeds: int = tap.arg(
        "--n-seeds",
        default=4,
        help="Number of seeded runs to average over for each trial.",
    )
    n_run_jobs: int = tap.arg(
        "--n-run-jobs",
        default=4,
        help="Number of runs of a same trial to execute in parallel.",
    )
    budget: int = tap.arg(
        "--budget",
        default=20,
        help="Target number of completed trials for each algorithm and layout directory.",
    )
    dry_run: bool = tap.arg(
        "--dry-run",
        default=False,
        help="Summarize existing studies and trials that would be started without changing anything.",
    )


def parse_pool_spec(pool_dir: Path) -> PoolSpec:
    """
    Validate a layout pool and derive its environment metadata.
    """
    if not pool_dir.is_dir():
        raise ValueError(f"Layout pool is not a directory: {pool_dir}")

    layout_files = sorted(path for path in pool_dir.iterdir() if path.is_file())
    required_layouts = TRAIN_POOL_SIZE + TEST_POOL_SIZE
    if len(layout_files) < required_layouts:
        raise ValueError(
            f"Layout pool {pool_dir} contains {len(layout_files)} files; at least {required_layouts} are required."
        )

    world = World.from_file(layout_files[0].as_posix())
    n_lasers = len(world.laser_sources)
    laser_label = "laser" if n_lasers == 1 else "lasers"
    layout_name = f"{world.width}x{world.height}_agents{world.n_agents}_{laser_label}{n_lasers}"
    layout_type = f"{pool_dir.name}-{layout_name}"
    return PoolSpec(path=pool_dir, time_limit=world.width * world.height, layout_type=layout_type)


def make_env(spec: PoolSpec, size: int, *, offset: int = 0) -> LLEPool:
    """
    Create a perspective-observation, flattened-state layout pool.
    """
    return LLEPool(
        spec.path,
        size,
        offset=offset,
        time_limit=spec.time_limit,
        obs_type="perspective",
        state_type="flattened",
    )


def hidden_sizes(trial: optuna.Trial, prefix: str) -> list[int]:
    """
    Suggest a uniform-width MLP architecture.
    """
    n_layers = trial.suggest_int(f"{prefix}.n_layers", 1, 5)
    size = trial.suggest_int(f"{prefix}.size", 32, 512, step=32)
    return [size] * n_layers


def suggest_train_policy(trial: optuna.Trial, noisy: bool, n_steps: int) -> Policy:
    """
    Suggest the exploration policy of a value-based trainer.

    Noisy networks explore through their own parameter noise, in which case the behaviour policy
    is greedy. Otherwise, epsilon is linearly annealed from 1.0 over a fraction of the training.

    @ai-generated
    """
    if noisy:
        return marl.policy.ArgMax()
    return marl.policy.EpsilonGreedy.linear(
        1.0,
        trial.suggest_categorical("epsilon_end", [0.01, 0.05]),
        int(trial.suggest_categorical("epsilon_decay_ratio", [0.1, 0.25, 0.5]) * n_steps),
    )


def suggest_target_updater(trial: optuna.Trial) -> TargetParametersUpdater:
    if trial.suggest_categorical("target_updater.__type__", ["soft", "hard"]) == "soft":
        return SoftUpdate(trial.suggest_float("target_updater.tau", 1e-3, 5e-2, log=True))
    return HardUpdate(trial.suggest_int("target_updater.update_period", 100, 1_000, step=10))


def make_dqn_trainer(
    trial: optuna.Trial,
    algo: Literal["vdn", "qmix", "dqn", "qplex"],
    env: EnvConfig[DiscreteMARLEnv],
    n_steps: int,
    catch_all: dict,
):
    """
    Build a trial-configured value-based trainer.

    The Q-network architecture is the one that `qnetworks.from_env` provides for the environment:
    only its duelling and noisy variants are searched over. The rest of the search space is kept
    deliberately narrow (coarse batch sizes, a fixed Adam optimiser and double Q-learning) so that
    a handful of trials is enough to locate promising regions. The replay memory is a
    `TransitionMemory` (guaranteed by the non-recurrent Q-network) whose capacity is searched over.

    @ai-generated
    """
    noisy = trial.suggest_categorical("qnetwork.noisy", [False, True])
    qnetwork = model_bank.qnetworks.from_env(
        env,
        recurrent=False,
        independent=True,
        duelling=trial.suggest_categorical("qnetwork.duelling", [True, False]),
        noisy=noisy,
    )
    shared_kwargs = {
        "qnetwork": qnetwork,
        "gamma": GAMMA,
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=8),
        "memory_size": trial.suggest_int("memory_size", 5_000, 100_000, step=1_000),
        "train_interval": (5, "step"),
        "target_updater": suggest_target_updater(trial),
        "optimiser_type": "adam",
        "double_qlearning": True,
        "train_policy": suggest_train_policy(trial, noisy, n_steps),
        "test_policy": marl.policy.ArgMax(),
        "ir_module": None,
        "vbe": None,
        "catch_all": catch_all,
    }
    match algo:
        case "dqn":
            return suggest(DQN, trial, mixer=None, **shared_kwargs)
        case "vdn":
            return suggest(VDN, trial, **shared_kwargs)
        case "qmix":
            return suggest(QMix, trial, mixer=mixers.QMix.from_env(env), **shared_kwargs)
    raise NotImplementedError()


def make_ppo_trainer(
    trial: optuna.Trial,
    algo: Literal["mappo", "ippo"],
    env: EnvConfig[DiscreteMARLEnv],
    n_steps: int,
    catch_all: dict,
):
    """
    Build a trial-configured PPO trainer.
    """
    sizes = hidden_sizes(trial, "actor_critic")
    actor, critic = model_bank.actor_critics.from_env(
        env,
        False,
        independent=True,
        actor_kwargs={"mlp_sizes": sizes},
        critic_kwargs={"mlp_sizes": sizes},
    )
    train_interval = trial.suggest_int("train_interval", 10, 250, step=10)
    minibatch_size = trial.suggest_int("minibatch_size", 5, train_interval)
    c2_type = trial.suggest_categorical("c2_type", ["linear", "constant"])
    if c2_type == "linear":
        c2 = Schedule.linear(
            start_value=trial.suggest_float("c2_start", 0.0, 1.0),
            end_value=trial.suggest_float("c2_end", 0.0, 1.0),
            n_steps=trial.suggest_int("c2_n_steps", 1, n_steps),
        )
    else:
        c2 = Schedule.constant(trial.suggest_float("c2", 0.0, 1.0))

    early_stopping_enabled = trial.suggest_categorical("early_stopping_enabled", [True, False])
    early_stopping_kl = (
        trial.suggest_float("early_stopping_kl", 1e-3, 0.1, log=True) if early_stopping_enabled else None
    )
    if algo == "mappo":
        mixer_name = trial.suggest_categorical("mixer", ["vdn", "qmix"])
        mixer = mixers.VDN.from_env(env) if mixer_name == "vdn" else mixers.QMix.from_env(env)
    else:
        mixer = None

    return PPO(
        actor,
        critic,
        mixer=mixer,
        lr_actor=trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True),
        lr_critic=trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True),
        train_interval=(train_interval, "step"),
        minibatch_size=minibatch_size,
        n_epochs=trial.suggest_int("n_epochs", 5, 20),
        eps_clip=trial.suggest_float("eps_clip", 0.01, 0.3),
        gae_lambda=0.95,
        c2=c2,
        normalize_advantages=trial.suggest_categorical("normalize_advantages", [True, False]),
        early_stopping_kl=early_stopping_kl,
    )


def make_trainer(trial: optuna.Trial, algo: Algo, env: EnvConfig[DiscreteMARLEnv], args: Args):
    """
    Build the requested trainer from an Optuna trial.
    """
    catch_all = {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": GAMMA}
    if algo in ("dqn", "vdn", "qmix", "qplex"):
        return make_dqn_trainer(trial, algo, env, args.n_steps, catch_all)
    return make_ppo_trainer(trial, algo, env, args.n_steps, catch_all)


def make_logdir(algo: Algo, spec: PoolSpec, trial: optuna.Trial) -> Path:
    """
    Build a fresh log directory for a trial.

    Trial numbers are only unique within a study, and a study that is restarted from scratch (or a
    trial that previously crashed) may leave a directory behind. Since `Experiment.create` refuses
    to overwrite an existing directory, a numbered suffix is appended until an unused path is found.

    @ai-generated
    """
    base = Path("logs", f"optuna-{algo}-{spec.layout_type}-{trial.number}")
    logdir = base
    attempt = 0
    while logdir.exists():
        attempt += 1
        logdir = base.with_name(f"{base.name}-retry{attempt}")
    return logdir


def objective(trial: optuna.Trial, algo: Algo, spec: PoolSpec, args: Args) -> float:
    """
    Train the seeded runs of a trial and return their mean final evaluation exit rate.

    @ai-generated
    """
    train_env = make_env(spec, TRAIN_POOL_SIZE)
    test_env = make_env(spec, TEST_POOL_SIZE, offset=TRAIN_POOL_SIZE)
    trainer = make_trainer(trial, algo, train_env, args)
    logdir = make_logdir(algo, spec, trial)
    trial.set_user_attr("logdir", logdir.as_posix())
    experiment = marl.Experiment.create(
        train_env,
        trainer,
        test_env=test_env,
        n_steps=args.n_steps,
        logdir=logdir,
    )
    experiment.run(
        seeds=args.n_seeds,
        save_weights=False,
        save_actions=False,
        test_interval=0,
        n_tests=TEST_POOL_SIZE,
        n_jobs=args.n_run_jobs,
        gpu_strategy=args.gpu_strategy,
        device_affinity=trial.number,
        disabled_gpus=args.disabled_gpus,
        quiet=True,
    )
    results = experiment.get_test_results(args.n_steps).select("mean-exit_rate").last().collect()
    if results.height == 0:
        raise RuntimeError(f"Trial {trial.number} produced no evaluation results (see {logdir}).")
    return results.item()


def tune(storage: JournalStorage, spec: PoolSpec, algo: Algo, args: Args):
    """
    Summarize or resume one study until its completed-trial budget is reached.
    """
    study_name = f"{algo.upper()}-{spec.layout_type}"
    if args.dry_run:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            LOGGER.info("[DRY RUN] %s: no existing study; would start %d trials.", study_name, args.budget)
            return
    else:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

    state_counts = {state: sum(trial.state == state for trial in study.trials) for state in optuna.trial.TrialState}
    completed = state_counts[optuna.trial.TrialState.COMPLETE]
    remaining = max(0, args.budget - completed)
    if args.dry_run:
        LOGGER.info(
            "[DRY RUN] %s: %d total (%d complete, %d failed, %d pruned, %d running, %d waiting); "
            "would start %d additional trials.",
            study_name,
            len(study.trials),
            completed,
            state_counts[optuna.trial.TrialState.FAIL],
            state_counts[optuna.trial.TrialState.PRUNED],
            state_counts[optuna.trial.TrialState.RUNNING],
            state_counts[optuna.trial.TrialState.WAITING],
            remaining,
        )
        return

    LOGGER.info("Study %s has %d/%d completed trials; scheduling %d.", study_name, completed, args.budget, remaining)
    remaining = args.budget - completed
    study.optimize(lambda trial: objective(trial, algo, spec, args), n_trials=remaining, n_jobs=args.n_jobs)
    if completed < args.budget:
        LOGGER.error("Giving up on %s with %d/%d completed trials.", study_name, completed, args.budget)

    if completed:
        LOGGER.info(
            "Best trial for %s: %d with value %s (%d/%d complete).",
            study_name,
            study.best_trial.number,
            study.best_value,
            completed,
            args.budget,
        )


def main(args: Args) -> None:
    """
    Tune every requested algorithm independently for each supplied layout directory.
    """
    if args.budget <= 0:
        raise ValueError(f"--budget must be a positive integer, got {args.budget}")

    specs = [parse_pool_spec(pool_dir) for pool_dir in args.pool_dirs]
    layout_types = [spec.layout_type for spec in specs]
    if len(layout_types) != len(set(layout_types)):
        raise ValueError("The supplied pool directories must have distinct layout types.")

    storage = JournalStorage(JournalFileBackend(STORAGE_FILE))
    for spec in specs:
        for algo in args.algos:
            try:
                tune(storage, spec, algo, args)
            except KeyboardInterrupt:
                raise
            except Exception:
                LOGGER.exception("Tuning failed for %s on %s.", algo.upper(), spec.path)


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("tuning.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception as error:
        LOGGER.error("Tuning command failed: %s", " ".join(sys.argv), exc_info=error)
        raise
