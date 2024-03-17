from .table_qlearning import VanillaQLearning, ReplayTableQLearning

from .mixers import QMix, VDN, Qatten, QPlex
from .dqn import DQN, RDQN
from .maic import MAICAlgo
from .rial import RIALAlgo


__all__ = [
    "VanillaQLearning",
    "ReplayTableQLearning",
    "QMix",
    "VDN",
    "Qatten",
    "QPlex",
    "DQN",
    "RDQN",
    "RIALAlgo",
    "MAICAlgo"
]
