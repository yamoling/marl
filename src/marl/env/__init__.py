"""
Set of toy environments for testing MARL algorithms.
"""

from .two_steps import TwoSteps, TwoStepsState
from .matrix_game import MatrixGame
from .coordinated_grid import CoordinatedGrid
from .wrappers import PotentialShaping
from .env_pool import EnvPool
from .connectn import ConnectN

__all__ = [
    "ConnectN",
    "TwoSteps",
    "TwoStepsState",
    "MatrixGame",
    "CoordinatedGrid",
    "EnvPool",
    "PotentialShaping",
]
