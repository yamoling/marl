from dataclasses import KW_ONLY, dataclass

from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models import Mixer
from marl.nn.mixers import VDN, QMix

from .config import Config


@dataclass
class MixerConfig[T: Mixer](Config[T]):
    _: KW_ONLY
    n_objectives: int = 1

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], **kwargs):
        return cls(n_objectives=env.n_objectives, **kwargs)


@dataclass
class VDNConfig(MixerConfig[VDN]):
    pass


@dataclass
class QMixConfig(MixerConfig[QMix]):
    n_agents: int
    state_size: int
    state_extras_size: int
    _: KW_ONLY
    embed_size: int = 64
    hypernet_embed_size: int = 64

    def make(self):
        return mixers.QMix(
            self.n_agents,
            self.state_size,
            self.state_extras_size,
            n_objectives=self.n_objectives,
            embed_size=self.embed_size,
            hypernet_embed_size=self.hypernet_embed_size,
        )
