"""
Lazy capability checks requiring a full deserialization with the current `marl` code.

This is the only module of the data layer that imports `marl` (lazily, inside the worker).
"""

import concurrent.futures
import re
import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from .cache import file_key
from .extract import CLASS_KEY
from .issues import Issue, Level, exception_detail
from .records import EXPERIMENT_FILE, ExperimentRecord

DEFAULT_TIMEOUT_S = 20.0

_UNKNOWN_SUBCLASS_RE = re.compile(r"Unknown subclass (?P<cls>[\w.]+) for (?P<base>[\w.]+)")
_MISSING_FIELD_RE = re.compile(r"Missing value for required field (?P<field>\w+) of class (?P<cls>[\w.]+)")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="studio-health")
_cache = dict[tuple, tuple[bool, list[Issue]]]()
_lock = threading.Lock()

Loader = Callable[[Path], Any]


@cache
def marl_version() -> str:
    """Version of the `marl` package plus the git HEAD of the project, computed once. @ai-generated"""
    from importlib.metadata import PackageNotFoundError, version

    try:
        parts = [version("marl")]
    except PackageNotFoundError:
        parts = ["unknown"]
    git = Path(__file__).resolve().parents[4] / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            head = (git / head.removeprefix("ref: ")).read_text().strip()
        parts.append(head)
    except OSError:
        pass
    return "+".join(parts)


def _load_experiment(path: Path) -> Any:
    """@ai-generated"""
    import marl

    return marl.Experiment.load(path)


def _find_class(node: Any, cls: str, prefix: str = "") -> str | None:
    """Dotted path of the first node (depth-first) whose class name is `cls`; "" for the root. @ai-generated"""
    if isinstance(node, dict):
        if node.get(CLASS_KEY) == cls:
            return prefix
        children = node.items()
    elif isinstance(node, list):
        children = ((str(i), v) for i, v in enumerate(node))
    else:
        return None
    for key, value in children:
        found = _find_class(value, cls, f"{prefix}.{key}" if prefix else key)
        if found is not None:
            return found
    return None


def error_path(exc: BaseException, raw: Any) -> str | None:
    """
    Best-effort dotted parameter path of a deserialization error, from the known messages
    "Unknown subclass X for Y" and "Missing value for required field f of class C".

    @ai-generated
    """
    message = str(exc)
    if match := _UNKNOWN_SUBCLASS_RE.search(message):
        found = _find_class(raw, match.group("cls"))
        return found or None
    if match := _MISSING_FIELD_RE.search(message):
        found = _find_class(raw, match.group("cls"))
        missing = match.group("field")
        if found is None:
            # The root may be tagged with another class (e.g. LightExperiment loaded as Experiment).
            return missing if isinstance(raw, dict) and missing not in raw else None
        return f"{found}.{missing}" if found else missing
    return None


def run_with_timeout[T](fn: Callable[[], T], timeout: float) -> T:
    """Run `fn` in a worker thread; raise `TimeoutError` after `timeout` seconds. @ai-generated"""
    future = _executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"Deserialization did not finish within {timeout:g} s") from exc


def check_launchable(
    record: ExperimentRecord, timeout: float = DEFAULT_TIMEOUT_S, loader: Loader = _load_experiment
) -> tuple[bool, list[Issue]]:
    """
    Full deserialization of experiment.json with the current marl code (`marl.Experiment.load`), in a
    worker thread with a timeout. Launch and replay require `params == "full"` and a successful load.
    Returns the verdict and the new issues (a `deserialize-failed` issue on failure). Results are cached
    by experiment.json stats and the marl version; timeouts are not cached.

    @ai-generated
    """
    exp_file = record.path / EXPERIMENT_FILE
    key = (str(record.path), file_key(exp_file), marl_version(), loader)
    with _lock:
        if key in _cache:
            ok, issues = _cache[key]
            return ok, list(issues)
    if not exp_file.is_file():
        return False, []
    try:
        run_with_timeout(lambda: loader(record.path), timeout)
    except TimeoutError as exc:
        issue = Issue(
            Level.ERROR, "deserialize-failed", "The experiment could not be loaded in time.", "experiment", None, exception_detail(exc)
        )
        return False, [issue]
    except Exception as exc:  # noqa: BLE001 - any failure of marl's deserialization means "not launchable"
        path = error_path(exc, record.raw)
        where = f" ({path})" if path else ""
        issue = Issue(
            Level.ERROR,
            "deserialize-failed",
            f"The experiment cannot be loaded with the current marl code{where}.",
            "experiment",
            path,
            exception_detail(exc),
        )
        result = (False, [issue])
    else:
        result = (record.capabilities.params == "full", [])
    with _lock:
        _cache[key] = result
    return result[0], list(result[1])


def clear_cache():
    """@ai-generated"""
    with _lock:
        _cache.clear()
