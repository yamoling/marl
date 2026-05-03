from marl.config import DQNConfig, ExperimentConfig, LLEConfig, MemoryConfig, PolicyConfig, QNetworkConfig
from marl.config.log_config import LogConfig
from marl.config.mixer_config import MixerConfig, QMixConfig, VDNConfig

env = LLEConfig(6)
qnet = QNetworkConfig.from_env(env)
config = ExperimentConfig(
    env,
    DQNConfig(
        qnet,
        PolicyConfig.epsilon("linear", 50_000, 0.01, 1),
        MemoryConfig("transition", 50_000),
    ),
    1_000_000,
)
config.create_runs(range(10), 5000, True, True)
json = config.to_json(beautify=True)
config.to_file("config.json", beautify=True)
print(json)
restored = ExperimentConfig.from_json(json)
print(restored)


exp = config.make()
