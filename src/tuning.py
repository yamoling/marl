import logging
import os
from typing import Literal

import dotenv
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import marl
from marl import policy
from marl.env import LLEConfig
from marl.nn.model_bank import qnetworks
from marl.utils.tuning import suggest

N_STEPS = 1_000


def objective(trial: optuna.Trial, algo: Literal["vdn", "qmix", "qplex", "maven", "mappo"]):
    env = LLEConfig(6, obs_type="layered", state_type="state")
    qnetwork = suggest(
        qnetworks.QCNN,
        trial,
        n_agents=env.n_agents,
        n_actions=env.n_actions,
        obs_shape=env.observation_shape,
        extras_shape=env.extras_shape,
    )
    match algo:
        case "vdn":
            mixer = marl.nn.mixers.VDN.from_env(env)
        case "qmix":
            mixer = marl.nn.mixers.QMix.from_env(env)
        case "qplex":
            mixer = marl.nn.mixers.QPlex.from_env(env)
        case other:
            raise NotImplementedError(f"Algorithm {other} not implemented yet.")
    trainer = suggest(marl.algos.DQN, trial, qnetwork=qnetwork, mixer=mixer, vbe=None, test_policy=policy.ArgMax(), gamma=0.95)
    exp = marl.Experiment(env, n_steps=N_STEPS, trainer=trainer, logdir=os.path.join("logs", f"optuna-{algo}-{trial.number}"))
    exp.run(
        seeds=5,
        save_weights=False,
        save_actions=False,
        test_interval=N_STEPS,
        disabled_gpus=range(6),
        n_jobs=5,
        gpu_strategy="scatter",
    )
    result = exp.get_results(5000)
    score = result["Test"].select("mean-exit_rate").last().collect().item()
    return score


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("tuning.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for algo in ("vdn", "qmix"):
        try:
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{algo.upper()}",
                storage=JournalStorage(JournalFileBackend("optuna_study.journal")),
                load_if_exists=True,
            )
            n_trials = 24
            study.optimize(lambda trial: objective(trial, algo=algo), n_trials=n_trials)  # type: ignore
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error("An error occurred during optimization.", exc_info=e)
