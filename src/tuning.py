import logging
import os
import random
from pathlib import Path
from typing import Literal

import dotenv
import optuna
from lle import ObservationType, World
from lle.env import LLE, SingleObjective
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

N_STEPS = 150_000
POOL_SIZE = 500
N_TRIALS = 74
N_JOBS = 6
MAP_DIR = Path("maps/5x5_agents2_lasers1")
POOL_DIR = Path("logs/optuna-map-pools")


def pool_dir(setting: Setting):
    dst = POOL_DIR / setting
    if dst.exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    groups = []
    for path in sorted((MAP_DIR / setting).iterdir()):
        world = World.from_file(path.as_posix())
        env = LLE(world, SingleObjective(world.n_agents), ObservationType.LAYERED, ObservationType.FLATTENED)
        for group in groups:
            if env.has_same_inouts(group[0][0]):
                group.append((env, path))
                break
        else:
            groups.append([(env, path)])
    maps = [path for _, path in max(groups, key=len)]
    # ponytail: source dirs mix incompatible I/O; pick the largest compatible group and cycle only if it has <1000 maps.
    for i in range(POOL_SIZE * 2):
        (dst / f"{i:04d}.txt").symlink_to(maps[i % len(maps)].resolve())
    return dst


def make_env(setting: Setting, *, offset: int = 0):
    return LLEPool(pool_dir(setting), POOL_SIZE, offset=offset, width=5, height=5, time_limit=25)


def hidden_sizes(trial: optuna.Trial, prefix: str):
    n_layers = trial.suggest_int(f"{prefix}.n_layers", 1, 5)
    size = trial.suggest_categorical(f"{prefix}.size", [64, 128, 256])
    return [size] * n_layers


def make_dqn_trainer(trial: optuna.Trial, algo: Literal["vdn", "qmix", "dqn"], env: EnvConfig[DiscreteMARLEnv], catch_all: dict):
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
            return suggest(DQN, trial, qnetwork=qnetwork, mixer=None, test_policy=test_policy, vbe=None, catch_all=catch_all)
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


def make_ppo_trainer(trial: optuna.Trial, algo: Literal["mappo", "ippo"], env: EnvConfig[DiscreteMARLEnv], catch_all: dict):
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
            n_steps=trial.suggest_int("c2_n_steps", 1, N_STEPS),
        )
    else:
        c2 = Schedule.constant(trial.suggest_float("c2", 0.0, 1.0))
    early_stopping_enabled = trial.suggest_categorical("early_stopping_enabled", [True, False])
    if early_stopping_enabled:
        es_kl = trial.suggest_float("early_stopping_kl", 1e-3, 0.1, log=True)
    else:
        es_kl = None
    if algo == "mappo":
        mixer_str = trial.suggest_categorical("mixer", ["vdn", "qmix"])
        mixer = mixers.VDN.from_env(env) if mixer_str == "vdn" else mixers.QMix.from_env(env)
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
        early_stopping_kl=es_kl,
    )


def make_trainer(trial: optuna.Trial, algo: Algo, env: EnvConfig[DiscreteMARLEnv]):
    catch_all = dict(n_agents=env.n_agents, n_actions=env.n_actions, gamma=0.99)
    if algo in ("dqn", "vdn", "qmix"):
        return make_dqn_trainer(trial, algo, env, catch_all)
    return make_ppo_trainer(trial, algo, env, catch_all)


def objective(trial: optuna.Trial, algo: Algo, setting: Setting):
    train_env = make_env(setting)
    test_env = make_env(setting, offset=500)
    trainer = make_trainer(trial, algo, train_env)
    exp = marl.Experiment.create(
        train_env,
        trainer,
        test_env=test_env,
        n_steps=N_STEPS,
        logdir=Path("logs", f"optuna-{algo}-{setting}-{trial.number}"),
    )
    exp.run(
        seeds=4,
        save_weights=False,
        save_actions=False,
        test_interval=N_STEPS,
        n_tests=POOL_SIZE,
        n_jobs=4,
        gpu_strategy="scatter",
        disabled_gpus=[2],
        quiet=True,
    )
    result = exp.get_results(N_STEPS)["Test"].select("mean-exit_rate").last().collect().item()
    return result


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("tuning.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    storage = JournalStorage(JournalFileBackend("optuna_study.journal"))
    for setting in ("cooperative", "independent"):
        for algo in ("vdn", "qmix", "mappo", "dqn", "ippo"):
            try:
                study = optuna.create_study(
                    direction="maximize",
                    study_name=f"{algo.upper()}-{setting}",
                    storage=storage,
                    load_if_exists=True,
                )
                n_performed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                remaining = N_TRIALS - n_performed
                if remaining > 0:
                    study.optimize(lambda trial: objective(trial, algo=algo, setting=setting), n_trials=remaining, n_jobs=N_JOBS)  # type: ignore
                print(f"Best trial for {algo.upper()}-{setting}: {study.best_trial.number} with value {study.best_value}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error("An error occurred during optimization.", exc_info=e)
