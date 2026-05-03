from marl import Experiment
from marl.config import DQNConfig, LLEConfig, MemoryConfig, PolicyConfig, QNetworkConfig
from marl.config.mixer_config import MixerConfig, QMixConfig, VDNConfig

env = LLEConfig(6)
qnet = QNetworkConfig.from_env(env)
config = Experiment(
    env,
    DQNConfig(
        qnet,
        PolicyConfig.epsilon("linear", 50_000, 0.01, 1),
        MemoryConfig("transition", 50_000),
    ),
    1_000_000,
)
json = config.to_json(beautify=True)
config.to_file("config.json", beautify=True)
print(json)
restored = Experiment.from_json(json)
print(restored)
