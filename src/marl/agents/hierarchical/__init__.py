from typing import TYPE_CHECKING

from .maven_agent import MAVENAgent

if TYPE_CHECKING:
    from marl.algos.haven import Haven


def __getattr__(name: str):
    """Load HAVEN lazily so agent package initialization does not import every algorithm. @ai-generated"""
    if name == "Haven":
        from marl.algos.haven import Haven

        return Haven
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Haven",
    "MAVENAgent",
]
