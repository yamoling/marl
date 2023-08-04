from .table_qlearning import VanillaQLearning, ReplayTableQLearning
from .qlearning import IDeepQLearning, IQLearning
from .vdn import LinearVDN, RecurrentVDN

from .rdqn import RDQN
from .mixed_dqn import LinearMixedDQN, RecurrentMixedDQN
from .mixers import QMix, VDN
from .dqn_nodes import DQN


__all__ = [
    "VanillaQLearning",
    "ReplayTableQLearning",
    "IDeepQLearning",
    "IQLearning",
    "LinearVDN",
    "RecurrentVDN",
    "RDQN",
    "LinearMixedDQN",
    "RecurrentMixedDQN",
    "QMix",
    "VDN",
    "DQN",
]
