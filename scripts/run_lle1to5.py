"""
Run MAPPO, IPPO, DQN (independent, no mixer), VDN and QMIX on LLE levels 1 to 5 (included) with
layered observations, for 1M time steps and 10 seeds.

The hyperparameters of every algorithm are those of the best trial of its study in the
certified-cooperation tuning journal (`tunings/certified-cooperation.journal`, studies named
`<ALGO>-<layout type>`). The trainers are rebuilt by replaying these parameters with a `FixedTrial`
into the search space that produced the journal (`scripts/tuning.py` at commit d2ea0990):

- MAPPO/IPPO: uniform-width actor/critic MLPs, PPO arguments and, for MAPPO, the VDN or QMIX mixer.
- DQN/VDN/QMIX: uniform-width Q-network, training policy, target updater and the other tuned trainer
  fields. The memory was not tuned and keeps a `TransitionMemory` of `MEMORY_SIZE` transitions.

All runs of all experiments share a single job queue. Existing experiments are reused and only
missing or incomplete seeds are launched. Incomplete runs restart from step zero, replacing their
partial metrics; completed and actively running seeds are left untouched.

Run with `uv run python scripts/run_lle1to5.py [options]` from the repository root.

@ai-generated
"""

import functools
import logging
from pathlib import Path
from typing import Literal, cast

import optuna
import torch
import typed_argparse as tap
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import FixedTrial

import marl
from marl import Experiment, algos
from marl.env import LLEConfig
from marl.models import Run, Trainer, TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.runners import parallel_run
from marl.utils import DeviceLike, Schedule
from marl.utils.tuning import suggest

logger = logging.getLogger(__name__)

LEVELS = [1, 2, 3, 4, 5]
Algo = Literal["mappo", "ippo", "dqn", "vdn", "qmix"]
ALGOS: tuple[Algo, ...] = ("mappo", "ippo", "dqn", "vdn", "qmix")
N_STEPS = 1_000_000
MEMORY_SIZE = 50_000
"""The certified-cooperation study did not tune the memory: DQN then defaulted to a 50k transition memory."""
GAMMA = 0.99
DEFAULT_JOURNAL = Path("tunings", "certified-cooperation.journal")
DEFAULT_LAYOUT_TYPE = "cooperative-9x9_agents3_lasers2"
"""The certified-cooperation layout whose best trials provide the hyperparameters."""


@functools.cache
def load_best_params(journal: Path, layout_type: str, algo: Algo) -> dict:
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
    n_complete = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if n_complete == 0:
        raise RuntimeError(f"Study {study_name!r} has no complete trials.")
    logger.info(f"Using best trial {study.best_trial.number} of {study_name} (value {study.best_value}, {n_complete} complete trials)")
    return study.best_params


def hidden_sizes(trial: optuna.Trial, prefix: str) -> list[int]:
    """Uniform-width MLP architecture of the certified-cooperation search space. @ai-generated"""
    return [trial.suggest_int(f"{prefix}.size", 32, 512, step=32)] * trial.suggest_int(f"{prefix}.n_layers", 1, 5)


def make_ppo_trainer(trial: optuna.Trial, algo: Literal["mappo", "ippo"], env: LLEConfig) -> algos.PPO:
    """
    Rebuild a PPO trainer in the search space of the certified-cooperation study
    (`make_ppo_trainer` of `scripts/tuning.py` at commit d2ea0990).

    @ai-generated
    """
    sizes = hidden_sizes(trial, "actor_critic")
    actor, critic = actor_critics.from_env(
        env,
        False,
        independent=True,
        actor_kwargs={"mlp_sizes": sizes},
        critic_kwargs={"mlp_sizes": sizes},
    )
    train_interval = trial.suggest_int("train_interval", 10, 250, step=10)
    minibatch_size = trial.suggest_int("minibatch_size", 5, train_interval)
    if trial.suggest_categorical("c2_type", ["linear", "constant"]) == "linear":
        c2 = Schedule.linear(
            start_value=trial.suggest_float("c2_start", 0.0, 1.0),
            end_value=trial.suggest_float("c2_end", 0.0, 1.0),
            n_steps=trial.suggest_int("c2_n_steps", 1, N_STEPS),
        )
    else:
        c2 = Schedule.constant(trial.suggest_float("c2", 0.0, 1.0))
    early_stopping_kl = None
    if trial.suggest_categorical("early_stopping_enabled", [True, False]):
        early_stopping_kl = trial.suggest_float("early_stopping_kl", 1e-3, 0.1, log=True)
    mixer = None
    if algo == "mappo":
        mixer_name = trial.suggest_categorical("mixer", ["vdn", "qmix"])
        mixer = mixers.VDN.from_env(env) if mixer_name == "vdn" else mixers.QMix.from_env(env)
    return algos.PPO(
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


def make_dqn_trainer(trial: optuna.Trial, algo: Literal["dqn", "vdn", "qmix"], env: LLEConfig) -> algos.DQN:
    """
    Rebuild a value-based trainer in the search space of the certified-cooperation study
    (`make_dqn_trainer` of `scripts/tuning.py` at commit d2ea0990).

    @ai-generated
    """
    qnetwork = qnetworks.from_env(
        env,
        independent=True,
        duelling=trial.suggest_categorical("qnetwork.duelling", [True, False]),
        noisy=trial.suggest_categorical("qnetwork.noisy", [False, True]),
        mlp_sizes=hidden_sizes(trial, "qnetwork"),
    )
    kwargs = {
        "qnetwork": qnetwork,
        "memory": TransitionMemory(MEMORY_SIZE),
        "test_policy": marl.policy.ArgMax(),
        "vbe": None,
        "catch_all": {"n_agents": env.n_agents, "n_actions": env.n_actions, "gamma": GAMMA},
    }
    match algo:
        case "dqn":
            return suggest(algos.DQN, trial, mixer=None, **kwargs)
        case "vdn":
            return suggest(algos.VDN, trial, **kwargs)
        case "qmix":
            return suggest(algos.QMix, trial, mixer=mixers.QMix.from_env(env), **kwargs)


def make_trainer(env: LLEConfig, algo: Algo, params: dict) -> Trainer:
    """Rebuild the trainer of `algo` from the tuned parameters of the journal. @ai-generated"""
    trial = cast(optuna.Trial, FixedTrial(params))
    if algo in ("mappo", "ippo"):
        return make_ppo_trainer(trial, algo, env)
    return make_dqn_trainer(trial, algo, env)


class Args(tap.TypedArgs):
    dry_run: bool = tap.arg("--dry-run", default=False, help="Only print what would be created/run.")
    no_training_data: bool = tap.arg(
        "--no-training-data", default=False, help="Do not log trainer-update metrics; retain training-episode and test metrics."
    )
    levels: list[int] = tap.arg("--levels", default=list(LEVELS), nargs="*", help=f"LLE levels to run (default: all, {list(LEVELS)}).")
    n_seeds: int = tap.arg("--n-seeds", default=10, help="Number of seeds per experiment (seeds 0..n-1).")
    device: DeviceLike = tap.arg("--device", default="auto", help='"auto", "cpu", "cuda:<id>" or a GPU index.')
    n_jobs: int | None = tap.arg("--n-jobs", default=None, help="Number of parallel run processes (defaults to the number of usable GPUs).")
    gpu_strategy: Literal["scatter", "group"] = tap.arg("--gpu-strategy", default="scatter")
    disabled_gpus: list[int] = tap.arg("--disabled-gpus", default=[], nargs="*")
    test_interval: int = tap.arg("--test-interval", default=10_000)
    n_tests: int = tap.arg("--n-tests", default=5)
    journal: Path = tap.arg("--journal", default=DEFAULT_JOURNAL, help="Optuna journal providing the hyperparameters.")
    layout_type: str = tap.arg("--layout-type", default=DEFAULT_LAYOUT_TYPE, help="Layout type of the studies to read.")


def collect_runs(args: Args) -> list[Run]:
    """
    Create missing experiments and queue missing or incomplete seeds, excluding active runs.
        Incomplete seeds restart from step zero rather than resuming from a checkpoint.

    @ai-generated
    """
    invalid = set(args.levels) - set(LEVELS)
    if invalid:
        raise ValueError(f"Invalid level(s) {sorted(invalid)}; available levels are {list(LEVELS)}.")
    runs = []
    for level in cast(list[Literal[1, 2, 3, 4, 5]], args.levels):
        for algo in ALGOS:
            name = f"lle{level}-{algo}"
            logdir = Path("logs") / name
            if logdir.exists():
                exp = Experiment.load(logdir)
                existing = {run.seed for run in exp.runs if run.is_running or run.is_complete}
            else:
                exp = None
                existing = set()
            seeds = [s for s in range(args.n_seeds) if s not in existing]
            if not seeds:
                logger.info(f"[skip] {name}: all seeds are complete or currently running.")
                continue
            if args.dry_run:
                if exp is None:
                    load_best_params(args.journal, args.layout_type, algo)
                logger.info(f"[dry-run] {name}: would run seeds {seeds} -> {logdir}")
                continue
            if exp is None:
                env = LLEConfig(level, obs_type="layered")
                params = load_best_params(args.journal, args.layout_type, algo)
                exp = Experiment.create(env, make_trainer(env, algo, params), logdir=name, n_steps=N_STEPS)
                logger.info(f"[new] Created experiment {exp.logdir} with parameters {params}")
            logger.info(f"{name}: queueing seeds {seeds}")
            runs += exp.create_runs(
                seeds,
                args.n_tests,
                args.test_interval,
                save_weights=False,
                save_actions=False,
                save_training_data=not args.no_training_data,
            )
    return runs


def main(args: Args):
    runs = collect_runs(args)
    if not runs:
        return
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = 1 if args.device == "cpu" else max(1, torch.cuda.device_count() - len(args.disabled_gpus))
    logger.info(f"Launching {len(runs)} run(s) with {n_jobs} parallel job(s).")
    parallel_run(
        runs,
        n_jobs=n_jobs,
        device=args.device,
        gpu_strategy=args.gpu_strategy,
        disabled_gpus=args.disabled_gpus,
        quiet=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tap.Parser(Args).bind(main).run()
