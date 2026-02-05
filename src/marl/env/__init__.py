"""
Set of toy environments for testing MARL algorithms.
"""

from .state_counter import StateCounter
from .reward_mask import NoReward

__all__ = [
    "StateCounter",
    "NoReward",
]
