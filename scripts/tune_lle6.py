import argparse
import logging
import math
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from marl import Experiment, algos
from marl.env import LLEConfig
from marl.models import Trainer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.policy import ArgMax, EpsilonGreedy
from marl.utils import Schedule

N_STEPS = 500_000
N_TRIALS = 20
N_RANDOM_TRIALS = 5
SEEDS = (0, 1, 2, 3)
Algo = Literal["mappo", "ippo", "dqn", "vdn", "qmix"]
LOGGER = logging.getLogger(__name__)


def make_env() -> LLEConfig:
    return LLEConfig(6, obs_type="layered", state_type="flattened")


def make_trainer(trial: optuna.Trial, algo: Algo, env: LLEConfig) -> Trainer:
    """Suggest baseline hyperparameters, keeping network architectures fixed. @ai-generated"""
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    grad_clip = trial.suggest_float("grad_norm_clipping", 1.0, 50.0)

    if algo in ("mappo", "ippo"):
        train_interval = trial.suggest_int("train_interval", 50, 100)
        actor, critic = actor_critics.from_env(env, recurrent=False, independent=False)
        return algos.PPO(
            actor,
            critic,
            mixer=mixers.VDN.from_env(env) if algo == "mappo" else None,
            train_interval=(train_interval, "step"),
            n_epochs=trial.suggest_int("n_epochs", 2, 10),
            minibatch_size=trial.suggest_int("minibatch_size", 8, train_interval // 2),
            lr_actor=trial.suggest_float("lr_actor", 5e-5, 3e-3, log=True),
            lr_critic=trial.suggest_float("lr_critic", 5e-5, 3e-3, log=True),
            eps_clip=trial.suggest_float("eps_clip", 0.1, 0.3),
            c1=Schedule.constant(trial.suggest_float("c1", 0.1, 2.0, log=True)),
            c2=Schedule.constant(trial.suggest_float("c2", 1e-4, 0.1, log=True)),
            gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            gamma=gamma,
            grad_norm_clipping=grad_clip,
            ir_module=None,
        )
    if algo in ("dqn", "vdn", "qmix"):
        qnetwork = qnetworks.from_env(env, recurrent=False, independent=True, duelling=False, noisy=False)
        kwargs: dict[str, Any] = {
            "memory": TransitionMemory(trial.suggest_int("memory_size", 5_000, 100_000, step=1_000)),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "train_interval": (5, "step"),
            "lr": trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            "target_updater": algos.SoftUpdate(trial.suggest_float("target_tau", 1e-3, 5e-2, log=True)),
            "train_policy": EpsilonGreedy.linear(
                1.0,
                trial.suggest_float("epsilon_end", 0.01, 0.1),
                trial.suggest_int("epsilon_steps", 25_000, 250_000, step=1_000),
            ),
            "test_policy": ArgMax(),
            "optimiser_type": trial.suggest_categorical("optimiser_type", ["adam", "rmsprop"]),
            "double_qlearning": True,
            "gamma": gamma,
            "grad_norm_clipping": grad_clip,
            "ir_module": None,
        }
        if algo == "dqn":
            return algos.DQN(qnetwork, **kwargs)
        if algo == "vdn":
            return algos.VDN(qnetwork, **kwargs)
        return algos.QMix(qnetwork, mixer=mixers.QMix.from_env(env), **kwargs)
    raise ValueError(f"Unsupported algorithm: {algo}")


def objective(trial: optuna.Trial, algo: Algo, args: argparse.Namespace) -> float:
    """Train four parallel replicas and validate their single final evaluations. @ai-generated"""
    if trial.number < 5:
        import time

        time.sleep(trial.number * 5)
    env = make_env()
    experiment = Experiment.create(
        env,
        make_trainer(trial, algo, env),
        n_steps=args.n_steps,
        logdir=Path("logs", args.name, algo, "score-0", f"trial-{trial.number:02d}"),
        loggers=("csv",),
    )
    trial.set_user_attr("logdir", experiment.logdir)
    trial.set_user_attr("training_seeds", list(SEEDS))
    experiment.run(
        seeds=SEEDS,
        n_jobs=args.n_experiment_jobs,
        n_tests=1,
        test_interval=-1,  # Disable periodic tests, including the test at step zero.
        save_weights=False,
        save_actions=False,
        save_training_data=False,
        device=args.device,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=True,
    )
    runs = sorted(experiment.runs, key=lambda run: run.seed)
    scores = []
    for run in runs:
        results = run.test_metrics.select("time_step", "score-0").collect()
        if not run.is_complete or results.height != 1 or results["time_step"][0] != args.n_steps:
            raise RuntimeError(f"Expected one evaluation at {args.n_steps} for seed {run.seed}; inspect {run.rundir}.")
        score = float(results["score-0"][0])
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite return for seed {run.seed}; inspect {run.rundir}.")
        scores.append(score)
    trial.set_user_attr("returns", scores)
    return mean(scores)


def tune(storage: JournalStorage, algo: Algo, args: argparse.Namespace) -> optuna.Study:
    """Run sequential TPE trials up to the persistent per-algorithm budget. @ai-generated"""
    study = optuna.create_study(
        study_name=f"{args.name}-{algo}-score-0",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=N_RANDOM_TRIALS, seed=args.sampler_seed),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    if any(trial.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING) for trial in study.trials):
        raise RuntimeError(f"Study {study.study_name} has unfinished trials. Reconcile them or use a fresh --name.")
    remaining = max(0, N_TRIALS - len(study.trials))
    if remaining:
        study.optimize(lambda trial: objective(trial, algo, args), n_trials=remaining, n_jobs=args.n_jobs)
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if completed:
        LOGGER.info("%s: best mean return (score-0) %.4f, parameters %s", study.study_name, study.best_value, study.best_params)
    return study


def dry_run(args: argparse.Namespace) -> None:
    """Read the journal and report the same remaining budget used by tuning. @ai-generated"""
    storage = JournalStorage(JournalFileBackend(args.storage.as_posix())) if args.storage.exists() else None
    for algo in args.algos:
        study_name = f"{args.name}-{algo}-score-0"
        trials = []
        if storage is not None:
            try:
                trials = optuna.load_study(study_name=study_name, storage=storage).trials
            except KeyError:
                LOGGER.info("%s: study not found in %s", study_name, args.storage)
        else:
            LOGGER.info("%s: journal does not exist: %s", study_name, args.storage)
        counts = {state.name: sum(trial.state == state for trial in trials) for state in optuna.trial.TrialState}
        remaining = max(0, N_TRIALS - len(trials))
        LOGGER.info(
            "%s: %d/%d trials recorded (%s); %d trials remaining = %d seeded runs (%d seeds per trial)",
            study_name,
            len(trials),
            N_TRIALS,
            ", ".join(f"{state}={count}" for state, count in counts.items()),
            remaining,
            remaining * len(SEEDS),
            len(SEEDS),
        )
        if counts["RUNNING"] or counts["WAITING"]:
            LOGGER.warning("%s: tuning is blocked by unfinished trials; reconcile them or use a fresh --name.", study_name)


def main() -> None:
    """Parse execution options and launch the requested independent studies. @ai-generated"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="lle6-layered-500k", help="Search identity and logs subdirectory; change for a fresh search.")
    parser.add_argument("--storage", type=Path, default=Path("tunings", "lle6-layered-500k-vdn-qmix-mappo.journal"))
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:<index> (auto requires an available GPU).")
    parser.add_argument("--gpu-strategy", choices=("scatter", "group"), default="scatter")
    parser.add_argument("--disabled-gpus", nargs="*", type=int, default=[])
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=N_STEPS, help="Training steps per seeded run (default: 500000).")
    parser.add_argument(
        "--algos", nargs="+", choices=("mappo", "ippo", "dqn", "vdn", "qmix"), default=["mappo", "ippo", "dqn", "vdn", "qmix"]
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of Optuna trials run in parallel per study.")
    parser.add_argument("--n-experiment-jobs", type=int, default=2, help="Number of seeded runs trained in parallel per trial.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Read the journal and report remaining trials and seeded runs without training."
    )
    args = parser.parse_args()
    if args.n_steps < 1:
        parser.error("--n-steps must be positive")
    if args.n_jobs < 1 or args.n_experiment_jobs < 1:
        parser.error("--n-jobs and --n-experiment-jobs must be positive")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name")
    logging.basicConfig(level=logging.INFO)
    if args.dry_run:
        dry_run(args)
        return
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(args.storage.as_posix()))
    for algo in args.algos:
        tune(storage, algo, args)


if __name__ == "__main__":
    main()
