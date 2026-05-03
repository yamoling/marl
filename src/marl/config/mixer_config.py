from dataclasses import KW_ONLY, dataclass

from marlenv import MARLEnv, MultiDiscreteSpace

from marl.models import Mixer
from marl.nn import mixers

from .config import Config


@dataclass
class MixerConfig(Config[Mixer]):
    n_agents: int
    n_actions: int
    state_size: int
    state_extras_size: int
    _: KW_ONLY
    n_objectives: int = 1

    @property
    def output_shape(self):
        if self.n_objectives == 1:
            return (self.n_actions,)
        return (self.n_actions, self.n_objectives)

    @classmethod
    def from_env(cls, env: MARLEnv[MultiDiscreteSpace], **kwargs):
        return cls(env.n_agents, env.n_actions, env.state_size, env.state_extras_size, n_objectives=env.n_objectives, **kwargs)


@dataclass
class VDNConfig(MixerConfig):
    def make(self):
        return mixers.VDN(self.output_shape, self.n_agents, n_objectives=self.n_objectives)


@dataclass
class QMixConfig(MixerConfig):
    _: KW_ONLY
    embed_size: int = 64
    hypernet_embed_size: int = 64

    def make(self):
        return mixers.QMix(
            self.output_shape,
            self.n_agents,
            self.state_size,
            self.state_extras_size,
            n_objectives=self.n_objectives,
            embed_size=self.embed_size,
            hypernet_embed_size=self.hypernet_embed_size,
        )
