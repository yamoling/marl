from .table_qlearning import VanillaQLearning, ReplayTableQLearning

from .mixers import QMix, VDN, Qatten
from .dqn import DQN, RDQN


__all__ = [
    "VanillaQLearning",
    "ReplayTableQLearning",
    "QMix",
    "VDN",
    "Qatten",
    "DQN",
    "RDQN",
]
