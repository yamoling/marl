"""
Small defensive extractors: raw JSON (dict) -> value or None.

None of these functions raise on malformed input; they return `None` instead.
"""

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CLASS_KEY = "class-name"


def get(raw: Any, *keys: str | int) -> Any:
    """Nested lookup that returns `None` on any missing key or wrong type. @ai-generated"""
    node = raw
    for key in keys:
        try:
            node = node[key]
        except (KeyError, TypeError, IndexError):
            return None
    return node


def as_int(value: Any) -> int | None:
    """@ai-generated"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def as_bool(value: Any) -> bool | None:
    """@ai-generated"""
    return value if isinstance(value, bool) else None


def as_str(value: Any) -> str | None:
    """@ai-generated"""
    return value if isinstance(value, str) else None


def class_name(node: Any) -> str | None:
    """@ai-generated"""
    return as_str(get(node, CLASS_KEY))


def algo(raw: Any) -> str | None:
    """
    Algorithm name: the trainer's class name, except for a plain DQN trainer with a mixer,
    where the mixer's class name is used (e.g. VDN, QMix).

    @ai-generated
    """
    trainer = get(raw, "trainer")
    name = class_name(trainer)
    if name == "DQN":
        mixer = class_name(get(trainer, "mixer"))
        if mixer is not None:
            return mixer
    return name


def env_name(raw: Any) -> str | None:
    """@ai-generated"""
    return as_str(get(raw, "env", "name"))


def test_env_name(raw: Any) -> str | None:
    """@ai-generated"""
    return as_str(get(raw, "test_env", "name"))


def n_steps(raw: Any) -> int | None:
    """@ai-generated"""
    return as_int(get(raw, "n_steps"))


def seed(raw: Any) -> int | None:
    """@ai-generated"""
    return as_int(get(raw, "seed"))


def loggers(raw: Any) -> list[str]:
    """@ai-generated"""
    value = get(raw, "loggers")
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def created(raw: Any, fallback: Path | None = None) -> str | None:
    """
    ISO-8601 creation time from `creation_timestamp`, falling back to the ctime of `fallback`.

    @ai-generated
    """
    value = as_str(get(raw, "creation_timestamp"))
    if value is not None:
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            pass
    if fallback is not None:
        try:
            return datetime.fromtimestamp(fallback.stat().st_ctime, tz=UTC).isoformat()
        except OSError:
            return None
    return None


def run_config(raw: Any) -> dict[str, Any]:
    """The run configuration fields exposed in `RunSummary.config`. @ai-generated"""
    return {
        "n_tests": as_int(get(raw, "n_tests")),
        "test_interval": as_int(get(raw, "test_interval")),
        "save_weights": as_bool(get(raw, "save_weights")),
        "save_actions": as_bool(get(raw, "save_actions")),
    }
