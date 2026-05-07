from dataclasses import dataclass, field

from marl.nn import mixers

from .dqn import DQN


@dataclass
class VDN(DQN[mixers.VDN]):
    mixer: mixers.VDN = field(default_factory=mixers.VDN)
