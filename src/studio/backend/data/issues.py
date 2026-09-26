"""Health model of the data layer: issues attached to experiments, runs and tables, and capabilities."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

ParamsCapability = Literal["full", "partial", "raw", "none"]
Health = Literal["ok", "warning", "error"]


class Level(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    level: Level
    code: str
    """Stable code, e.g. `missing-table` (see backend.md §2)."""
    message: str
    """One human sentence."""
    scope: str
    """`experiment`, `run:<run-id>` or `table:<run-id>/<table>`."""
    path: str | None = None
    """File path relative to the experiment, or a dotted parameter path."""
    detail: str | None = None
    """Exception type and message (no traceback)."""

    def to_json(self) -> dict[str, Any]:
        """@ai-generated"""
        return {
            "level": str(self.level),
            "code": self.code,
            "message": self.message,
            "scope": self.scope,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass
class Capabilities:
    metrics: bool
    params: ParamsCapability
    replay: bool | None = None
    """None means that the lazy check has not been computed yet."""
    launch: bool | None = None

    def to_json(self) -> dict[str, Any]:
        """@ai-generated"""
        return {"metrics": self.metrics, "params": self.params, "replay": self.replay, "launch": self.launch}


def exception_detail(exc: BaseException) -> str:
    """@ai-generated"""
    return f"{type(exc).__name__}: {exc}"


def issue_counts(issues: Iterable[Issue]) -> dict[str, int]:
    """Count issues per level, with every level present in the result. @ai-generated"""
    counts = {str(level): 0 for level in Level}
    for issue in issues:
        counts[str(issue.level)] += 1
    return counts


def health_of(issues: Iterable[Issue]) -> Health:
    """Maximal level of the given issues, where `info` counts as `ok`. @ai-generated"""
    levels = {issue.level for issue in issues}
    if Level.ERROR in levels:
        return "error"
    if Level.WARNING in levels:
        return "warning"
    return "ok"
