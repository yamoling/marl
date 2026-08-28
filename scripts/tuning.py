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
from marl.algos import DQN, PPO, VDN, QMix
from marl.env import EnvConfig, LLEPool
from marl.nn import mixers, model_bank
from marl.utils import Schedule
from marl.utils.tuning import suggest

Algo = Literal["vdn", "qmix", "dqn", "mappo", "ippo"]
Setting = Literal["cooperative", "independent"]

ALGOS: tuple[Algo, ...] = ("vdn", "qmix", "mappo", "dqn", "ippo")
TRAIN_POOL_SIZE = 5_000
N_EVALUATION_EPISODES = 1_000
N_SEEDS = 4
N_RUN_JOBS = 4

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolSpec:
    path: Path
    time_limit: int
    layout_type: str


class Args(tap.TypedArgs):
    pool_dirs: list[Path] = tap.arg(
        positional=True,
        nargs="+",
        help="One or more directories containing at least 6,000 layouts (5,000 training and 1,000 evaluation).",
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
    required_layouts = TRAIN_POOL_SIZE + N_EVALUATION_EPISODES
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


def make_dqn_trainer(
    trial: optuna.Trial,
    algo: Literal["vdn", "qmix", "dqn"],
    env: EnvConfig[DiscreteMARLEnv],
    catch_all: dict,
):
    """
    Build a trial-configured value-based trainer.
    """
    qnetwork = model_bank.qnetworks.from_env(
        env,
        independent=True,
        duelling=trial.suggest_categorical("qnetwork.duelling", [True, False]),
        noisy=trial.suggest_categorical("qnetwork.noisy", [False, True]),
        mlp_sizes=hidden_sizes(trial, "qnetwork"),
    )
    test_policy = marl.policy.ArgMax()
    match algo:
        case "dqn":
            return suggest(
                DQN, trial, qnetwork=qnetwork, mixer=None, test_policy=test_policy, vbe=None, catch_all=catch_all
            )
        case "vdn":
            return suggest(VDN, trial, qnetwork=qnetwork, test_policy=test_policy, vbe=None, catch_all=catch_all)
        case "qmix":
            return suggest(
                cls=QMix,
                trial=trial,
                qnetwork=qnetwork,
                mixer=mixers.QMix.from_env(env),
                test_policy=test_policy,
                vbe=None,
                catch_all=catch_all,
            )


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
    catch_all = {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": 0.99}
    if algo in ("dqn", "vdn", "qmix"):
        return make_dqn_trainer(trial, algo, env, catch_all)
    return make_ppo_trainer(trial, algo, env, args.n_steps, catch_all)


def objective(trial: optuna.Trial, algo: Algo, spec: PoolSpec, args: Args) -> float:
    """
    Train four seeded runs and return their mean final evaluation exit rate.
    """
    train_env = make_env(spec, TRAIN_POOL_SIZE)
    test_env = make_env(spec, N_EVALUATION_EPISODES, offset=TRAIN_POOL_SIZE)
    trainer = make_trainer(trial, algo, train_env, args)
    experiment = marl.Experiment.create(
        train_env,
        trainer,
        test_env=test_env,
        n_steps=args.n_steps,
        logdir=Path("logs", f"optuna-{algo}-{spec.layout_type}-{trial.number}"),
    )
    experiment.run(
        seeds=N_SEEDS,
        save_weights=False,
        save_actions=False,
        test_interval=0,
        n_tests=N_EVALUATION_EPISODES,
        n_jobs=N_RUN_JOBS,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=True,
    )
    return experiment.get_test_results(args.n_steps).select("mean-exit_rate").last().collect().item()


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
    if remaining:
        study.optimize(lambda trial: objective(trial, algo, spec, args), n_trials=remaining, n_jobs=args.n_jobs)

    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
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
    Tune every supported algorithm independently for each supplied layout directory.
    """
    if args.budget <= 0:
        raise ValueError(f"--budget must be a positive integer, got {args.budget}")

    specs = [parse_pool_spec(pool_dir) for pool_dir in args.pool_dirs]
    layout_types = [spec.layout_type for spec in specs]
    if len(layout_types) != len(set(layout_types)):
        raise ValueError("The supplied pool directories must have distinct layout types.")

    storage = JournalStorage(JournalFileBackend("optuna_study.journal"))
    for spec in specs:
        for algo in ALGOS:
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
