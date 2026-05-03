from marl import Experiment
from marl.config import DQNConfig, LLEConfig, MemoryConfig, PolicyConfig, QNetworkConfig
from marl.config.mixer_config import MixerConfig, QMixConfig, VDNConfig

env = LLEConfig(6)
qnet = QNetworkConfig.from_env(env)
experiment = Experiment(
    env,
    DQNConfig(
        qnet,
        PolicyConfig.epsilon("linear", 50_000, 0.01, 1),
        MemoryConfig("transition", 50_000),
    ),
    2_000,
)

experiment.save()
experiment.run(test_interval=500)
