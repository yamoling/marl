"""
Set of toy environments for testing MARL algorithms.
"""

from .two_steps import TwoSteps, TwoStepsState
from .matrix_game import MatrixGame
from .coordinated_grid import CoordinatedGrid
from .wrappers import CurriculumLearning, ZeroPunishment, PotentialShaping
from .env_pool import EnvPool

__all__ = [
    "TwoSteps",
    "TwoStepsState",
    "MatrixGame",
    "CoordinatedGrid",
    "CurriculumLearning",
    "ZeroPunishment",
    "EnvPool",
    "PotentialShaping",
]
