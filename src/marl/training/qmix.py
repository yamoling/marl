from dataclasses import dataclass

from marl.nn import mixers

from .dqn import DQN


@dataclass
class QMix(DQN[mixers.QMix]): ...
