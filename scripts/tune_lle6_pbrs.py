"""
Hyperparameter tuning of MASER, QMIX, and HAVEN on LLE level 6 with potential-based reward shaping (PBRS).

The search spaces are those of `scripts/tuning.py` (`make_maser_trainer`, `make_dqn_trainer`, and
`make_haven_trainer`) and the environment is the PBRS level 6 of `scripts/lle6-pbrs.py`. Each algorithm has its
own study, tuned one after the other. Each trial trains `--n-seeds` seeded runs and is scored by their mean final
exit rate: the shaped return is not comparable across shaping variants, while the exit rate is.

Usage:
    uv run python scripts/tune_lle6_pbrs.py --algos maser qmix haven --n-jobs 2 --n-run-jobs 4
    uv run python scripts/tune_lle6_pbrs.py --dry-run
"""

import logging
import os
import sys
from pathlib import Path
from typing import Literal, get_args

import dotenv
import optuna
import tuning
import typed_argparse as tap
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import marl
from marl.env import LLEConfig
from marl.models import Trainer
from marl.utils import DeviceLike

LOGGER = logging.getLogger(__name__)

Algo = Literal["maser", "qmix", "haven"]
ALGOS: tuple[Algo, ...] = get_args(Algo)


class Args(tap.TypedArgs):
    algos: list[Algo] = tap.arg("--algos", nargs="+", default=list(ALGOS), help="Algorithms to tune, one study each.")
    storage: Path = tap.arg("--storage", default=Path("tunings", "lle6-pbrs.journal"), help="Optuna journal file.")
    pbrs: Literal["individual", "joint"] = tap.arg("--pbrs", default="joint", help="PBRS subgoal extras.")
    pbrs_subgoals: Literal["laser", "crossing"] = tap.arg("--pbrs-subgoals", default="crossing", help="PBRS subgoal kind.")
    n_steps: int = tap.arg("--n-steps", default=500_000, help="Training steps per seeded run.")
    n_seeds: int = tap.arg("--n-seeds", default=4, help="Seeded runs averaged by each trial.")
    budget: int = tap.arg("--budget", default=50, help="Target number of completed trials per algorithm.")
    n_jobs: int = tap.arg("--n-jobs", default=1, help="Optuna trials run in parallel.")
    n_run_jobs: int = tap.arg("--n-run-jobs", default=4, help="Seeded runs of a trial run in parallel.")
    device: DeviceLike = tap.arg("--device", default="auto", help='"auto", "cpu", "cuda:<gpu_id>", or an int index.')
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", nargs="*", default=[])
    dry_run: bool = tap.arg("--dry-run", default=False, help="Report the state of the studies without training.")

    def study_name(self, algo: Algo) -> str:
        return f"lle6-pbrs-{self.pbrs}-{self.pbrs_subgoals}-{algo}-{self.n_steps}"


def make_env(args: Args, algo: Algo) -> LLEConfig:
    """HAVEN's subgoals are passed through extras that the environment pads. @ai-generated"""
    padding = tuning.HAVEN_N_SUBGOALS if algo == "haven" else None
    return LLEConfig(6, pbrs=args.pbrs, pbrs_subgoals=args.pbrs_subgoals, agent_id=True, extra_padding_size=padding)


def make_trainer(trial: optuna.Trial, algo: Algo, env: LLEConfig, n_steps: int) -> Trainer:
    """Suggest a trainer in the search space of `scripts/tuning.py`. @ai-generated"""
    catch_all = {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": tuning.GAMMA}
    match algo:
        case "maser":
            return tuning.make_maser_trainer(trial, env, n_steps, catch_all)
        case "qmix":
            return tuning.make_dqn_trainer(trial, "qmix", env, n_steps, catch_all)
        case "haven":
            return tuning.make_haven_trainer(trial, env, n_steps, catch_all)
    raise ValueError(f"Unsupported algorithm: {algo}")


def objective(trial: optuna.Trial, algo: Algo, args: Args) -> float:
    """
    Train the seeded runs of a trial and return their mean final exit rate.

    Logs are kept (without weights nor actions) so that every trial remains inspectable.

    @ai-generated
    """
    env = make_env(args, algo)
    experiment = marl.Experiment.create(
        env,
        make_trainer(trial, algo, env, args.n_steps),
        n_steps=args.n_steps,
        logdir=Path("logs", "tuning", args.study_name(algo), f"trial-{trial.number:03d}"),
        loggers=("csv",),
    )
    trial.set_user_attr("logdir", str(experiment.logdir))
    experiment.run(
        seeds=args.n_seeds,
        n_tests=1,  # LLE and the ArgMax test policy are deterministic.
        test_interval=-1,  # Only evaluate at the end of training.
        save_weights=False,
        save_actions=False,
        save_training_data=False,
        n_jobs=args.n_run_jobs,
        gpu_strategy=args.gpu_strategy,
        device=args.device,
        device_affinity=trial.number,
        disabled_gpus=args.disabled_gpus,
        quiet=True,
        limit_torch_threads=None,
    )
    results = experiment.get_test_results(args.n_steps).select("mean-exit_rate").last().collect()
    if results.height == 0:
        raise RuntimeError(f"Trial {trial.number} produced no evaluation results (see {experiment.logdir}).")
    return float(results.item())


def tune(storage: JournalStorage, algo: Algo, args: Args):
    """Create or resume the study of an algorithm and run trials until its budget is reached. @ai-generated"""
    study_name = args.study_name(algo)
    if args.dry_run:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            study = None
    else:
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
    trials = [] if study is None else study.trials
    counts = {state.name: sum(trial.state == state for trial in trials) for state in optuna.trial.TrialState}
    remaining = max(0, args.budget - counts["COMPLETE"])
    LOGGER.info("%s: %s; %d/%d trials remaining.", study_name, counts, remaining, args.budget)
    if study is None or args.dry_run or remaining == 0:
        return
    study.optimize(lambda trial: objective(trial, algo, args), n_trials=remaining, n_jobs=args.n_jobs)
    if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        LOGGER.info("%s: best trial %d, exit rate %.4f with %s", study_name, study.best_trial.number, study.best_value, study.best_params)


def main(args: Args):
    """Tune each requested algorithm in turn; the failure of one study does not stop the others. @ai-generated"""
    if min(args.budget, args.n_steps, args.n_seeds, args.n_jobs, args.n_run_jobs) <= 0:
        raise ValueError("--budget, --n-steps, --n-seeds, --n-jobs and --n-run-jobs must be positive.")
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(args.storage.as_posix()))
    for algo in dict.fromkeys(args.algos):
        try:
            tune(storage, algo, args)
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Tuning failed for %s.", algo.upper())


if __name__ == "__main__":
    dotenv.load_dotenv()
    logging.basicConfig(
        handlers=[logging.FileHandler("lle6_pbrs_tuning.log", mode="a"), logging.StreamHandler()],
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        tap.Parser(Args).bind(main).run()
    except KeyboardInterrupt:
        raise
    except Exception:
        LOGGER.exception(f"Tuning on LLE level 6 with PBRS failed with command line '{sys.argv}'.")
        raise
