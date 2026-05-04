from marl import Experiment
from marl.config import LLEConfig
from marl.models import TransitionMemory
from marl.nn import mixers
from marl.nn.model_bank import qnetworks
from marl.policy import EpsilonGreedy
from marl.training import DQN

env = LLEConfig(6)
experiment = Experiment(
    env,
    DQN(
        qnetworks.from_env(env),
        EpsilonGreedy.linear(50_000, 0.01, 1),
        TransitionMemory(50_000),
        mixer=mixers.QPlex.from_env(env),
    ),
    2_000,
)

experiment.save()
experiment.run(test_interval=500)
