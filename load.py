from marl import Experiment
from marl.env import LLEConfig
from marl.models import TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import actor_critics, qnetworks
from marl.policy import EpsilonGreedy
from marl.training import DQN, MAVEN, PPO

env = LLEConfig(6, maven_noise_size=16)
experiment = Experiment(
    env,
    MAVEN(
        qnetworks.from_env(env),
        EpsilonGreedy.linear(50_000, 0.01, 1),
        16,
        env.n_actions,
        TransitionMemory(50_000),
        mixer=mixers.QMixMAVEN.from_env(env),
    ),
    2_000,
)

experiment = Experiment(
    env,
    PPO(
        actor_critics.from_env(env),
        mixers.VDN.from_env(env),
    ),
    2_000,
)


experiment.save()
experiment.run(test_interval=500)
