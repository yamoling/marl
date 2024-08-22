from typing import Literal
import marl
import marlenv
from lle import LLE, ObservationType
from marl.algo import mixers


def mappo(env: marlenv.RLEnv) -> tuple[marl.RLAlgo, marl.Trainer]:
    nn = marl.nn.model_bank.CNN_ActorCritic.from_env(env)
    algo = marl.algo.PPO(
        ac_network=nn,
        train_policy=marl.policy.CategoricalPolicy(),
    )
    trainer = marl.training.PPOTrainer(
        network=nn,
        memory=marl.models.TransitionMemory(20),
    )
    return algo, trainer


def dqn_with_mixer(env: marlenv.RLEnv, mixer_str: Literal["vdn", "qmix", "qplex"]):
    qnetwork = marl.nn.model_bank.CNN.from_env(env)
    train_policy = marl.policy.EpsilonGreedy.constant(0.1)

    algo = marl.algo.DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
    )
    match mixer_str:
        case "vdn":
            mixer = mixers.VDN(env.n_agents)
        case "qmix":
            mixer = mixers.QMix.from_env(env)
        case "qplex":
            mixer = mixers.QPlex.from_env(env)
        case _:
            raise ValueError()

    trainer = marl.training.DQNTrainer(
        qnetwork=qnetwork,
        train_policy=train_policy,
        memory=marl.models.TransitionMemory(5_000),
        mixer=mixer,
        train_interval=(1, "step"),
        lr=5e-4,
    )
    return algo, trainer


if __name__ == "__main__":
    env = env = LLE.level(6).obs_type(ObservationType.LAYERED).build()
    env = marlenv.Builder(env).time_limit(env.width * env.height // 2).agent_id().build()

    # algo, trainer = dqn_with_mixer(env, "vdn")
    algo, trainer = mappo(env)
    runner = marl.Runner(env, algo, trainer)
    runner.run(logdir="logs/multi-agent-example", n_steps=10_000, test_interval=1000, n_tests=10)
