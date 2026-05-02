import lle

from marl.config import PolicyConfig
from marl.config.mixer_config import MixerConfig, QMixConfig, VDNConfig

env = lle.level(6).builder().time_limit(50).build()
qmix = QMixConfig.from_env(env, hypernet_embed_size=400)
vdn = VDNConfig.from_env(env)
qmix.make()
ser = qmix.to_json()
restored = MixerConfig.from_json(ser)
mix2 = restored.make()
print(mix2)


policies = list[PolicyConfig](
    [
        PolicyConfig.argmax(),
        PolicyConfig.epsilon("constant", 0.5),
        PolicyConfig.softmax(5),
    ]
)
for p in policies:
    json = p.to_json()
    print(p, json)
    deser = PolicyConfig.from_json(json)
    print(deser)
    print("Reconstructed policy: ", deser.make())
