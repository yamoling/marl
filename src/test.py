from marl import Experiment
from marl.env import LLEConfig
from marl.nn.model_bank import qnetworks
from marl.policy import EpsilonGreedy
from marl.training import MAVEN

env = LLEConfig(6, maven_noise_size=16)
experiment = Experiment(
    env,
    MAVEN(
        qnetworks.from_env(env),
        EpsilonGreedy.linear(50_000, 0.01, 1),
        env,
    ),
    2_000,
)

experiment.run(test_interval=500)
