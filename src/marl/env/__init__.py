"""
Set of toy environments for testing MARL algorithms.
"""

from .two_steps import TwoSteps, TwoStepsState
from .matrix_game import MatrixGame
from .coordinated_grid import CoordinatedGrid
from .lle_curriculum import CurriculumLearning
from .env_pool import EnvPool
from .zero_punishment import ZeroPunishment

__all__ = [
    "TwoSteps",
    "TwoStepsState",
    "MatrixGame",
    "CoordinatedGrid",
    "CurriculumLearning",
    "ZeroPunishment",
    "EnvPool",
]
