import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import marl
import logging
import dotenv
from marlenv import MARLEnv, MultiDiscreteSpace
from marl.nn.model_bank import qnetworks
from typing import cast, Literal, Any
import lle
import os

N_STEPS = 1_000_000
SHAPING = False


def objective(trial: optuna.Trial, algo: Literal["vdn", "qmix", "qplex", "maven", "mappo"]):
    env = lle.level(6).obs_type("layered").state_type("state")
    if SHAPING:
        env = env.pbrs(gamma=1.0, reward_value=1.0, lasers_to_reward=[(4, 0), (6, 12)])
    env = env.builder().agent_id().time_limit(78).build()
    match algo:
        case "vdn":
            dqn_params = suggest_dqn(trial, env)
            trainer = marl.training.DQN(mixer=marl.nn.mixers.VDN.from_env(env), **dqn_params)
        case "qmix":
            dqn_params = suggest_dqn(trial, env)
            trainer = marl.training.DQN(mixer=marl.nn.mixers.QMix.from_env(env), **dqn_params)
        case "qplex":
            dqn_params = suggest_dqn(trial, env)
            trainer = marl.training.DQN(mixer=marl.nn.mixers.QPlex.from_env(env), **dqn_params)
        case other:
            raise NotImplementedError(f"Algorithm {other} not implemented yet.")
    exp = marl.Experiment.create(
        env,
        n_steps=N_STEPS,
        trainer=trainer,
        save_weights=False,
        save_actions=False,
        test_interval=N_STEPS,
        logdir=os.path.join("logs", f"optuna-{algo}-{trial.number}"),
        replace_if_exists=True,
    )
    exp.run(4, "scatter", quiet=True, n_parallel=4, n_tests=30, device="auto")
    result = exp.get_experiment_results()
    df = result["Test"]
    score = df.select("mean-exit_rate").last().collect().item()
    return score


def suggest_dqn(trial: optuna.Trial, env: MARLEnv[MultiDiscreteSpace]) -> dict[str, Any]:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    gamma = trial.suggest_float("gamma", 0.9, 1.0, step=0.01)
    epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.001, 0.1, log=True)
    epsilon_decay = trial.suggest_int("epsilon_decay", 1000, 500_000, step=1000)
    grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.5, 50.0, step=0.5)
    memory_size = trial.suggest_int("memory_size", 5_000, 100_000, step=5_000)
    optimizer_type = trial.suggest_categorical("optimizer_type", ["adam", "rmsprop"])
    double_qlearning = trial.suggest_categorical("double_qlearning", [True, False])
    update_type = trial.suggest_categorical("update_type", ["soft", "hard"])
    mlp_size_1 = trial.suggest_int("mlp_size_1", 64, 512, step=64)
    mlp_size_2 = trial.suggest_int("mlp_size_2", 64, 512, step=64)
    if update_type == "soft":
        tau = trial.suggest_float("tau", 1e-3, 1e-1, log=True)
        target_updater = marl.training.qtarget_updater.SoftUpdate(tau)
    else:
        target_update_interval = trial.suggest_int("target_update_interval", 50, 1_000, step=50)
        target_updater = marl.training.qtarget_updater.HardUpdate(target_update_interval)

    return dict(
        qnetwork=qnetworks.QCNN.from_env(env, mlp_sizes=(mlp_size_1, mlp_size_2)),
        train_policy=marl.policy.EpsilonGreedy.linear(epsilon_start, epsilon_end, epsilon_decay),
        memory=marl.models.TransitionMemory(memory_size),
        optimiser_type=cast(Literal["adam", "rmsprop"], optimizer_type),
        gamma=gamma,
        batch_size=batch_size,
        lr=learning_rate,
        grad_norm_clipping=grad_norm_clipping,
        target_updater=target_updater,
        double_qlearning=double_qlearning,
        test_policy=marl.policy.ArgMax(),
    )


if __name__ == "__main__":
    import time

    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("tuning.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        # Sleep for two hours
        time.sleep(5400)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"QMIX - {'shaping' if SHAPING else 'no shaping'}",
            storage=JournalStorage(JournalFileBackend("optuna_study.journal")),
            load_if_exists=True,
        )
        study.optimize(lambda trial: objective(trial, algo="qmix"), n_trials=24, n_jobs=8)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("An error occurred during optimization.", exc_info=e)
