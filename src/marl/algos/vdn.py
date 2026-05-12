from dataclasses import KW_ONLY, dataclass, field

from marl.nn import mixers

from .dqn import DQN


@dataclass(unsafe_hash=True)
class VDN(DQN[mixers.VDN]):
    _: KW_ONLY
    mixer: mixers.VDN = field(default_factory=mixers.VDN)

    def __post_init__(self):
        assert isinstance(self.mixer, mixers.VDN)
        return super().__post_init__()
