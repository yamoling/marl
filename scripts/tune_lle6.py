import logging
import os
import sys
from pathlib import Path
from typing import Literal

import dotenv
import optuna
import typed_argparse as tap
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import marl
from marl.algos import ACER, DQN, LAN, PPO, VDN, QMix, QPlex
from marl.env import EnvConfig, LLEConfig
from marl.models import Trainer
from marl.nn import mixers, model_bank
from marl.nn.lan import LANValue, LocalAdvantageNetwork
from marl.utils import Schedule
from marl.utils.tuning import suggest

Algo = Literal["vdn", "qmix", "dqn", "qplex", "lan", "mappo", "ippo", "acer"]

ALGOS: tuple[Algo, ...] = ("vdn", "qmix", "dqn", "qplex", "lan", "mappo", "ippo", "acer")
LLE_LEVEL = 6
LLE_TIME_LIMIT = 78
GAMMA = 0.99
LOGGER = logging.getLogger(__name__)


class Args(tap.TypedArgs):
    algos: list[Algo] = tap.arg(
        "--algos",
        nargs="*",
        default=list(ALGOS),
        help="Algorithms to tune.",
    )
    n_jobs: int = tap.arg(
        "--n-jobs",
        default=8,
        help="Number of parallel Optuna jobs.",
    )
    n_steps: int = tap.arg(
        "--n-steps",
        default=1_000_000,
        help="Number of training steps per trial.",
    )
    seeds: int = tap.arg(
        "--seeds",
        default=4,
        help="Number of seeds averaged for each trial.",
    )
    run_n_jobs: int = tap.arg(
        "--run-n-jobs",
        default=4,
        help="Number of parallel seeded runs launched for each trial.",
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
        default=50,
        help="Target number of completed trials for each algorithm.",
    )
    storage_file: str = tap.arg(
        "--storage-file",
        default="lvl6-tuning.journal",
        help="Optuna journal storage file.",
    )
    dry_run: bool = tap.arg(
        "--dry-run",
        default=False,
        help="Summarize existing studies and trials that would be started without changing anything.",
    )


def make_dqn_family_trainer(
    trial: optuna.Trial,
    algo: Literal["vdn", "qmix", "dqn", "qplex"],
    env: EnvConfig,
    catch_all: dict,
):
    qnetwork = model_bank.qnetworks.from_env(
        env,
        independent=True,
        duelling=trial.suggest_categorical("qnetwork.duelling", [True, False]),
        noisy=False,
    )
    test_policy = marl.policy.ArgMax()
    gamma = GAMMA
    match algo:
        case "dqn":
            return suggest(
                DQN,
                trial,
                qnetwork=qnetwork,
                mixer=None,
                test_policy=test_policy,
                vbe=None,
                gamma=gamma,
                catch_all=catch_all,
            )
        case "vdn":
            return suggest(
                VDN, trial, qnetwork=qnetwork, test_policy=test_policy, vbe=None, gamma=gamma, catch_all=catch_all
            )
        case "qmix":
            return suggest(
                cls=QMix,
                trial=trial,
                qnetwork=qnetwork,
                mixer=mixers.QMix.from_env(env),
                test_policy=test_policy,
                vbe=None,
                gamma=gamma,
                catch_all=catch_all,
            )
        case "qplex":
            return suggest(
                cls=QPlex,
                trial=trial,
                qnetwork=qnetwork,
                mixer=mixers.QPlex.from_env(env, n_actions=env.n_actions),
                test_policy=test_policy,
                vbe=None,
                gamma=gamma,
                catch_all=catch_all,
            )


def make_lan_trainer(trial: optuna.Trial, env: EnvConfig, catch_all: dict):
    mean_center = trial.suggest_categorical("lan.mean_center", [True, False])
    qnetwork = LocalAdvantageNetwork.from_env(env, mean_center=mean_center)
    value_network = LANValue(
        env.observation_shape,
        env.extras_shape,
        env.state_shape,
        env.state_extra_shape,
        qnetwork.hidden_size,
    )
    return suggest(
        LAN,
        trial,
        qnetwork=qnetwork,
        value_network=value_network,
        mixer=None,
        vbe=None,
        gamma=GAMMA,
        catch_all=catch_all,
    )


def make_ppo_trainer(
    trial: optuna.Trial,
    algo: Literal["mappo", "ippo"],
    env: EnvConfig,
    n_steps: int,
    catch_all: dict,
):
    actor, critic = model_bank.actor_critics.from_env(env, False, independent=True)
    train_interval = trial.suggest_int("train_interval", 10, 156, step=2)
    minibatch_size = trial.suggest_int("minibatch_size", 5, min(30, train_interval))
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
        mixer = mixers.VDN.from_env(env)
    else:
        mixer = None
    return PPO(
        actor,
        critic,
        mixer=mixer,
        gamma=GAMMA,
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


def make_acer_trainer(trial: optuna.Trial, env: EnvConfig, catch_all: dict):
    actor = model_bank.actor_critics.discrete_actors.from_env(env, False, independent=True)
    critic = model_bank.qnetworks.from_env(
        env,
        independent=True,
        duelling=trial.suggest_categorical("acer_critic.duelling", [True, False]),
        noisy=False,
    )
    return suggest(ACER, trial, actor=actor, critic=critic, mixer=None, gamma=GAMMA, catch_all=catch_all)


def make_trainer(trial: optuna.Trial, algo: Algo, env: EnvConfig, args: Args):
    catch_all = {"n_agents": env.n_agents, "n_actions": env.n_actions}
    if algo in ("vdn", "qmix", "dqn", "qplex"):
        return make_dqn_family_trainer(trial, algo, env, catch_all)
    if algo == "lan":
        return make_lan_trainer(trial, env, catch_all)
    if algo == "acer":
        return make_acer_trainer(trial, env, catch_all)
    return make_ppo_trainer(trial, algo, env, args.n_steps, catch_all)


def resume_or_create_experiment(trial: optuna.Trial, algo: Algo, env: EnvConfig, trainer: Trainer, args: Args):
    """
    Return the experiment of `trial` together with the seeds that still have to be trained.

    An interrupted trial leaves its experiment directory behind with some of its runs already
    finished. In that case, the experiment is loaded instead of being re-created and only the
    seeds whose run did not reach `args.n_steps` are scheduled again.

    @ai-generated
    """
    logdir = Path("logs", f"optuna-lvl6-{algo}-{trial.number}")
    requested_seeds = list(range(args.seeds))
    if not logdir.exists():
        return marl.Experiment.create(env, trainer, n_steps=args.n_steps, logdir=logdir), requested_seeds
    experiment = marl.Experiment.load(logdir)
    if experiment.n_steps != args.n_steps or experiment.trainer.to_json() != trainer.to_json():
        raise FileExistsError(
            f"Experiment directory {logdir} already exists but holds a different configuration: "
            "delete it (or the stale logs of the previous study) before resuming this study."
        )
    complete_seeds = {run.seed for run in experiment.runs if run.is_complete}
    remaining = [seed for seed in requested_seeds if seed not in complete_seeds]
    LOGGER.info(
        "Resuming %s: %d/%d runs already complete, scheduling seeds %s.",
        logdir,
        args.seeds - len(remaining),
        args.seeds,
        remaining,
    )
    return experiment, remaining


def objective(trial: optuna.Trial, algo: Algo, args: Args) -> float:
    """
    Train `args.seeds` seeded runs and return their mean final sum of rewards.

    Runs that are already complete from a previous, interrupted attempt at this trial are
    reused as-is, so that only the missing seeds are trained.

    Testing is disabled during training (`test_interval=0`), but the runner always tests
    once at the very last time step, so the single resulting test episode's score is the
    final performance of the trained agent for each seed.
    """
    env = LLEConfig(LLE_LEVEL, obs_type="perspective", state_type="flattened", time_limit=LLE_TIME_LIMIT)
    trainer = make_trainer(trial, algo, env, args)
    experiment, seeds = resume_or_create_experiment(trial, algo, env, trainer, args)
    if len(seeds) > 0:
        experiment.run(
            seeds=seeds,
            n_jobs=args.run_n_jobs,
            gpu_strategy=args.gpu_strategy,
            test_interval=0,
            save_weights=False,
            save_actions=False,
            disabled_gpus=args.disabled_gpus,
            quiet=True,
        )
    else:
        LOGGER.info("All %d runs of %s are already complete: reusing their results.", args.seeds, experiment.logdir)
    return experiment.get_test_results(args.n_steps).select("mean-score-0").last().collect().item()


def tune(storage: JournalStorage, algo: Algo, args: Args):
    """
    Summarize or resume one study until its completed-trial budget is reached.

    @ai-generated
    """
    study_name = f"{algo.upper()}-LLE-lvl{LLE_LEVEL}"
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
        study.optimize(lambda trial: objective(trial, algo, args), n_trials=remaining, n_jobs=args.n_jobs)

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
    Tune every requested algorithm on LLE level 6.

    @ai-generated
    """
    if args.budget <= 0:
        raise ValueError(f"--budget must be a positive integer, got {args.budget}")

    storage = JournalStorage(JournalFileBackend(args.storage_file))
    for algo in args.algos:
        try:
            tune(storage, algo, args)
        except KeyboardInterrupt:
            raise
        except Exception:
            LOGGER.exception("Tuning failed for %s.", algo.upper())


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("tune_lle6.log", mode="a"), logging.StreamHandler()],
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
