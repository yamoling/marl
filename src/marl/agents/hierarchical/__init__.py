from typing import TYPE_CHECKING

from .maven_agent import MAVENAgent

if TYPE_CHECKING:
    from marl.algos.haven import HavenAgent


def __getattr__(name: str):
    """Load HAVEN lazily so agent package initialization does not import every algorithm. @ai-generated"""
    if name == "HavenAgent":
        from marl.algos.haven import HavenAgent

        return HavenAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HavenAgent",
    "MAVENAgent",
]
