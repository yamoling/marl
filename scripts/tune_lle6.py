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
Algo = Literal["mappo", "ippo", "vdn", "qmix"]
LOGGER = logging.getLogger(__name__)


def make_env() -> LLEConfig:
    return LLEConfig(6, obs_type="layered", state_type="flattened")


def make_trainer(trial: optuna.Trial, algo: Algo, env: LLEConfig) -> Trainer:
    """Suggest baseline hyperparameters, keeping network architectures fixed. @ai-generated"""
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    grad_clip = trial.suggest_float("grad_norm_clipping", 1.0, 20.0, log=True)
    if algo in ("mappo", "ippo"):
        actor, critic = actor_critics.from_env(env, recurrent=False, independent=True)
        return algos.PPO(
            actor,
            critic,
            mixer=mixers.VDN.from_env(env) if algo == "mappo" else None,
            train_interval=(trial.suggest_int("train_interval", 50, 100), "step"),
            n_epochs=trial.suggest_int("n_epochs", 2, 10),
            minibatch_size=trial.suggest_categorical("minibatch_size", [8, 16, 32]),
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
    if algo in ("vdn", "qmix"):
        qnetwork = qnetworks.from_env(env, recurrent=False, independent=True, duelling=False, noisy=False)
        kwargs: dict[str, Any] = {
            "memory": TransitionMemory(trial.suggest_int("memory_size", 5_000, 100_000, step=5_000)),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "train_interval": (trial.suggest_categorical("train_interval", [1, 5, 10]), "step"),
            "lr": trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            "target_updater": algos.SoftUpdate(trial.suggest_float("target_tau", 1e-3, 5e-2, log=True)),
            "train_policy": EpsilonGreedy.linear(
                1.0,
                trial.suggest_float("epsilon_end", 0.01, 0.1),
                trial.suggest_int("epsilon_steps", 25_000, 250_000, step=25_000),
            ),
            "test_policy": ArgMax(),
            "optimiser_type": trial.suggest_categorical("optimiser_type", ["adam", "rmsprop"]),
            "double_qlearning": True,
            "gamma": gamma,
            "grad_norm_clipping": grad_clip,
            "ir_module": None,
        }
        if algo == "vdn":
            return algos.VDN(qnetwork, **kwargs)
        return algos.QMix(qnetwork, mixer=mixers.QMix.from_env(env), **kwargs)
    raise ValueError(f"Unsupported algorithm: {algo}")


def objective(trial: optuna.Trial, algo: Algo, args: argparse.Namespace) -> float:
    """Train four parallel replicas and validate their single final evaluations. @ai-generated"""
    env = make_env()
    experiment = Experiment.create(
        env,
        make_trainer(trial, algo, env),
        n_steps=N_STEPS,
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
        if not run.is_complete or results.height != 1 or results["time_step"][0] != N_STEPS:
            raise RuntimeError(f"Expected one evaluation at {N_STEPS} for seed {run.seed}; inspect {run.rundir}.")
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


def main() -> None:
    """Parse execution options and launch the requested independent studies. @ai-generated"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="lle6-layered-500k", help="Search identity and logs subdirectory; change for a fresh search.")
    parser.add_argument("--storage", type=Path, default=Path("tunings", "lle6-layered-500k.journal"))
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:<index> (auto requires an available GPU).")
    parser.add_argument("--gpu-strategy", choices=("scatter", "group"), default="scatter")
    parser.add_argument("--disabled-gpus", nargs="*", type=int, default=[])
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--algos", nargs="+", choices=("mappo", "ippo", "vdn", "qmix"), default=["mappo", "ippo", "vdn", "qmix"])
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of Optuna trials run in parallel per study.")
    parser.add_argument("--n-experiment-jobs", type=int, default=2, help="Number of seeded runs trained in parallel per trial.")
    args = parser.parse_args()
    if args.n_jobs < 1 or args.n_experiment_jobs < 1:
        parser.error("--n-jobs and --n-experiment-jobs must be positive")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name")
    logging.basicConfig(level=logging.INFO)
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(args.storage.as_posix()))
    for algo in args.algos:
        tune(storage, algo, args)


if __name__ == "__main__":
    main()
