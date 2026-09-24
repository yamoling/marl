from typing import TYPE_CHECKING

from .option_agent import OptionAgent
from .qlearning import DQNAgent, QAgent
from .random_agent import RandomAgent, RandomOneHot
from .replay_agent import ReplayAgent
from .simple_agent import ContinuousAgent, DiscreteAgent, DiscreteOneHotAgent, SimpleAgent

if TYPE_CHECKING:
    from marl.algos.haven import HavenAgent


def __getattr__(name: str):
    """Load the HAVEN compatibility export only when requested. @ai-generated"""
    if name == "HavenAgent":
        from marl.algos.haven import HavenAgent

        return HavenAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ContinuousAgent",
    "DQNAgent",
    "DiscreteAgent",
    "DiscreteOneHotAgent",
    "HavenAgent",
    "OptionAgent",
    "QAgent",
    "RandomAgent",
    "RandomOneHot",
    "ReplayAgent",
    "SimpleAgent",
]
