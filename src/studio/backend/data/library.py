"""
Entry point of the data layer: discovery, cached records and every read-only query of the API,
returning plain JSON-ready data in the shapes of api-contract.md.
"""

import math
import os
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..errors import conflict
from . import health as health_checks
from .cache import FileKey, file_key
from .issues import Issue, health_of
from .metrics import build_catalog
from .params import ParamRow, SearchContext, flatten, matches, parse_query
from .records import (
    EXPERIMENT_FILE,
    PID_FILE,
    ExperimentRecord,
    Fingerprint,
    PidChecker,
    build_experiment,
    check_pid,
    discover,
    is_experiment_dir,
    scan_fingerprint,
)
from .series import SeriesQuery, batch_to_json, compute, compute_batch
from .sources import scan
from .sources.base import TIME_STEP, TIMESTAMP
from .sources.sqlite import SUFFIXES as SQLITE_SUFFIXES

TEST_TABLE = "test"
LISTING_REVALIDATE_S = 10.0
HOT_WINDOW_S = 120.0


def quick_key(path: Path) -> tuple:
    """
    Cheap validity key of an experiment for listings: stats of experiment.json and databases, and the
    mtimes of the experiment and run directories (which change when files are created or removed,
    e.g. a pid file or a new table, but not when a file is appended to).

    @ai-generated
    """
    parts: list[Any] = [file_key(path / EXPERIMENT_FILE)]
    try:
        parts.append(os.stat(path).st_mtime_ns)
        with os.scandir(path) as it:
            for e in it:
                if e.name.endswith(SQLITE_SUFFIXES) or e.is_dir():
                    st = e.stat()
                    parts.append((e.name, st.st_mtime_ns, st.st_size))
    except OSError:
        return ()
    return tuple(sorted(parts[2:])) + tuple(parts[:2])


@dataclass(frozen=True)
class QuickState:
    key: tuple
    hot: bool
    checked: float
    """`time.monotonic()` of the last full fingerprint check."""

    @classmethod
    def after_full_check(cls, path: Path, fp: Fingerprint, hot_window_s: float) -> "QuickState":
        """
        An experiment is hot if a run has a pid file or a file modified within `hot_window_s` (e.g. a
        table being appended to by a process without pid file).

        @ai-generated
        """
        recent = (time.time() - hot_window_s) * 1e9
        hot = any(PID_FILE in files or any(k is not None and k[1] > recent for k in files.values()) for files in fp.files.values())
        return cls(quick_key(path), hot, time.monotonic())


_experiment_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="studio-experiments")


def _allow_list(value: str | Iterable[str] | None) -> set[str] | None:
    """Comma-separated string or iterable -> casefolded set; None/empty means no filter. @ai-generated"""
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    result = {item.strip().casefold() for item in items if item.strip()}
    return result or None


def _json_value(value: Any) -> Any:
    """@ai-generated"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Library:
    def __init__(
        self, root: Path | str | None = None, pid_checker: PidChecker = check_pid, health_timeout: float = health_checks.DEFAULT_TIMEOUT_S
    ):
        if root is None:
            from ..settings import logs_root

            root = logs_root()
        self.root = Path(root).resolve()
        self.roots = (self.root,)
        self.pid_checker = pid_checker
        self.health_timeout = health_timeout
        self._records = dict[str, ExperimentRecord]()
        self._params = dict[str, tuple[tuple, list[ParamRow]]]()
        self._health = dict[str, tuple[FileKey, bool, list[Issue]]]()
        self._quick = dict[str, QuickState]()
        self._lock = threading.RLock()
        self.revalidate_s = LISTING_REVALIDATE_S
        """Maximal age of the last full fingerprint check of an experiment when listing."""
        self.hot_window_s = HOT_WINDOW_S
        """Files modified less than this ago make an experiment "hot" (always fully checked)."""

    def set_roots(self, roots: Sequence[Path | str]):
        """Replace the loaded roots and clear caches before exposing the new workspace. @ai-edited"""
        resolved = tuple(Path(root).resolve() for root in roots)
        self.roots = resolved
        self.root = resolved[0] if resolved else self.root
        self.invalidate()

    def root_for(self, path: Path) -> Path:
        """Return the configured root containing a discovered experiment path. @ai-generated"""
        for root in self.roots:
            if path.is_relative_to(root):
                return root
        raise ValueError("Experiment is outside loaded logdirs")

    def _paths(self) -> dict[str, Path]:
        """Discover all roots, rejecting ambiguous relative IDs rather than picking one. @ai-generated"""
        paths: dict[str, Path] = {}
        for root in self.roots:
            for path in discover(root):
                experiment_id = path.relative_to(root).as_posix()
                if experiment_id in paths:
                    raise conflict(f"Experiment ID {experiment_id} exists in multiple logdirs", "duplicate-experiment-id")
                paths[experiment_id] = path
        return paths

    # ------------------------------------------------------------ Records

    def resolve(self, experiment_id: str) -> Path | None:
        """
        Directory of an experiment id, confined to the logs root (no absolute paths, `..`, backslashes
        or symlink escapes). None if invalid or not an experiment.

        @ai-generated
        """
        if not isinstance(experiment_id, str) or not experiment_id or "\x00" in experiment_id or "\\" in experiment_id:
            return None
        relative = Path(experiment_id)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        matches = []
        for root in self.roots:
            path = (root / relative).resolve()
            if path != root and path.is_relative_to(root) and is_experiment_dir(path):
                matches.append(path)
        if len(matches) > 1:
            raise conflict(f"Experiment ID {experiment_id} exists in multiple logdirs", "duplicate-experiment-id")
        return matches[0] if matches else None

    def ids(self) -> list[str]:
        """@ai-generated"""
        return list(self._paths())

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """The record of an experiment, rebuilt only when its fingerprint changes. @ai-generated"""
        path = self.resolve(experiment_id)
        if path is None:
            return None
        return self._get_path(path)

    def _get_path(self, path: Path) -> ExperimentRecord:
        """Record of an already validated experiment directory. @ai-generated"""
        cached, fp = self._cached(path)
        return cached if cached is not None else self._build(path, fp)

    def _cached(self, path: Path) -> tuple[ExperimentRecord | None, Fingerprint]:
        """The cached record if its (full) fingerprint is unchanged, and the current fingerprint. @ai-generated"""
        fp = scan_fingerprint(path, self.root_for(path), self.pid_checker)
        self._quick[self._id_of(path)] = QuickState.after_full_check(path, fp, self.hot_window_s)
        with self._lock:
            record = self._records.get(self._id_of(path))
            if record is not None and record.mtime_key == fp.key:
                return self._apply_health(record), fp
        return None, fp

    def _cached_for_listing(self, path: Path) -> tuple[ExperimentRecord | None, Fingerprint | None]:
        """
        Like `_cached`, but trusts the cached record without stat-ing every table file when the
        directory mtimes (entries created or removed) and experiment.json/database stats are unchanged,
        the experiment is not "hot" (pid file, or a file modified recently at the last full check), and
        the last full check is less than `revalidate_s` old. The fingerprint is None in that case.

        @ai-generated
        """
        experiment_id = self._id_of(path)
        state = self._quick.get(experiment_id)
        if state is not None and not state.hot and time.monotonic() - state.checked < self.revalidate_s:
            with self._lock:
                record = self._records.get(experiment_id)
            if record is not None and quick_key(path) == state.key:
                with self._lock:
                    return self._apply_health(record), None
        return self._cached(path)

    def _build(self, path: Path, fp: Fingerprint) -> ExperimentRecord:
        """@ai-generated"""
        record = build_experiment(path, self.root_for(path), self.pid_checker, fp)
        with self._lock:
            self._records[record.id] = record
            return self._apply_health(record)

    def _id_of(self, path: Path) -> str:
        return path.relative_to(self.root_for(path)).as_posix()

    def records(self) -> list[ExperimentRecord]:
        """
        Records of every discovered experiment. Fingerprints are computed serially (cheap, GIL-bound);
        stale records are rebuilt in parallel (file reads and Polars release the GIL).

        @ai-generated
        """
        checked = [(path, *self._cached_for_listing(path)) for path in self._paths().values()]
        stale = [(path, fp) for path, record, fp in checked if record is None and fp is not None]
        built = iter(_experiment_pool.map(lambda item: self._build(*item), stale))
        return [record if record is not None else next(built) for _, record, _ in checked]

    def invalidate(self, experiment_id: str | None = None):
        """Forget cached records (all of them if no id is given). @ai-generated"""
        with self._lock:
            if experiment_id is None:
                self._records.clear()
                self._params.clear()
                self._health.clear()
                self._quick.clear()
            else:
                self._records.pop(experiment_id, None)
                self._params.pop(experiment_id, None)
                self._health.pop(experiment_id, None)
                self._quick.pop(experiment_id, None)

    def _apply_health(self, record: ExperimentRecord) -> ExperimentRecord:
        """Apply a stored health check if it still matches experiment.json. @ai-generated"""
        stored = self._health.get(record.id)
        if stored is None:
            return record
        key, ok, issues = stored
        if key != file_key(record.path / EXPERIMENT_FILE):
            del self._health[record.id]
            record.capabilities.launch = record.capabilities.replay = None
            record.issues[:] = [i for i in record.issues if i.code != "deserialize-failed"]
            return record
        record.capabilities.launch = record.capabilities.replay = ok
        record.issues.extend(i for i in issues if i not in record.issues)
        return record

    # ------------------------------------------------------------ Queries

    def param_rows(self, record: ExperimentRecord) -> list[ParamRow]:
        """@ai-generated"""
        with self._lock:
            cached = self._params.get(record.id)
            if cached is not None and cached[0] == record.mtime_key:
                return cached[1]
        rows = flatten(record.raw)
        with self._lock:
            self._params[record.id] = (record.mtime_key, rows)
        return rows

    def _context(self, record: ExperimentRecord) -> SearchContext:
        """@ai-generated"""
        s = record.summary
        return SearchContext(record.id, s.name, s.algo, s.env_name, s.test_env_name, record.status, self.param_rows(record))

    def list_summaries(
        self,
        q: str = "",
        algo: str | Iterable[str] | None = None,
        status: str | Iterable[str] | None = None,
        health: str | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        `ExperimentSummary[]` matching the query and the allow-lists, sorted by creation date
        (most recent first, unknown dates last).

        @ai-generated
        """
        algos, statuses, healths = _allow_list(algo), _allow_list(status), _allow_list(health)
        terms = parse_query(q)
        result = list[dict[str, Any]]()
        for record in self.records():
            if algos is not None and (record.summary.algo or "").casefold() not in algos:
                continue
            if statuses is not None and record.status.casefold() not in statuses:
                continue
            if healths is not None and health_of(record.all_issues) not in healths:
                continue
            if terms and not matches(terms, self._context(record)):
                continue
            result.append(record.to_summary_json())
        dated = sorted((s for s in result if s["created"]), key=lambda s: s["created"], reverse=True)
        return dated + [s for s in result if not s["created"]]

    def detail(self, experiment_id: str) -> dict[str, Any] | None:
        """The API's `ExperimentDetail`. @ai-generated"""
        record = self.get(experiment_id)
        if record is None:
            return None
        return record.to_summary_json() | {
            "raw": record.raw,
            "issues": [i.to_json() for i in record.all_issues],
            "capabilities": record.capabilities.to_json(),
            "runs": [run.to_json() for run in record.runs],
            "params": [row.to_json() for row in self.param_rows(record)],
        }

    def health(self, experiment_id: str) -> dict[str, Any] | None:
        """
        Run the lazy capability check (full deserialization) and return
        `{capabilities, issues}` with all the issues after the check.

        @ai-generated
        """
        record = self.get(experiment_id)
        if record is None:
            return None
        ok, issues = health_checks.check_launchable(record, self.health_timeout)
        with self._lock:
            self._health[record.id] = (file_key(record.path / EXPERIMENT_FILE), ok, issues)
            self._apply_health(record)
        return {"capabilities": record.capabilities.to_json(), "issues": [i.to_json() for i in record.all_issues]}

    def catalog(self, experiment_id: str) -> dict[str, Any] | None:
        """@ai-generated"""
        record = self.get(experiment_id)
        return build_catalog(record).to_json() if record is not None else None

    def series(self, queries: Sequence[SeriesQuery | dict]) -> list[dict[str, Any]]:
        """`[{ok: true, result} | {ok: false, issue}]` in the order of the queries. @ai-generated"""
        return batch_to_json(compute_batch(self.get, queries))

    def params(self, ids: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Flattened parameters per experiment; unknown ids are omitted. @ai-generated"""
        result = dict[str, list[dict[str, Any]]]()
        for experiment_id in ids:
            record = self.get(experiment_id)
            if record is not None:
                result[record.id] = [row.to_json() for row in self.param_rows(record)]
        return result

    def preview(self, experiment_id: str, points: int = 60) -> dict[str, Any] | None:
        """Mean and ci95 of the default metric, with at most `points` points. @ai-generated"""
        record = self.get(experiment_id)
        if record is None:
            return None
        metric = build_catalog(record).default_metric
        if metric is None:
            return {"metric": None, "result": None}
        query = SeriesQuery(record.id, metric.table, metric.metric, max_points=max(4, points))
        return {"metric": metric.to_json(), "result": compute(record, query).to_json()}

    def test_steps(self, experiment_id: str) -> list[int] | None:
        """Sorted union over runs of the time steps of the `test` table. @ai-generated"""
        record = self.get(experiment_id)
        if record is None:
            return None
        steps = set[int]()
        for run in record.runs:
            table = run.tables.get(TEST_TABLE)
            if table is None or TIME_STEP not in table.columns:
                continue
            try:
                values = scan(table).select(pl.col(TIME_STEP).cast(pl.Float64, strict=False)).drop_nulls().unique().collect()
            except (pl.exceptions.PolarsError, OSError):
                continue
            steps.update(int(v) for v in values[TIME_STEP].to_list() if math.isfinite(v))
        return sorted(steps)

    def episodes(self, experiment_id: str, step: int) -> list[dict[str, Any]] | None:
        """
        Test episodes of every run at `step`: `{run, seed, test, step, metrics, has_actions}`, where
        `test` is the index of the episode within the step.

        @ai-generated
        """
        record = self.get(experiment_id)
        if record is None:
            return None
        result = list[dict[str, Any]]()
        for run in record.runs:
            table = run.tables.get(TEST_TABLE)
            if table is None or TIME_STEP not in table.columns:
                continue
            try:
                rows = scan(table).filter(pl.col(TIME_STEP).cast(pl.Float64, strict=False) == step).collect()
            except (pl.exceptions.PolarsError, OSError):
                continue
            has_actions = (run.path / "test" / str(step) / "actions.json").is_file()
            for test, row in enumerate(rows.drop(TIME_STEP, TIMESTAMP, strict=False).iter_rows(named=True)):
                result.append(
                    {
                        "run": run.id,
                        "seed": run.seed,
                        "test": test,
                        "step": step,
                        "metrics": {k: _json_value(v) for k, v in row.items()},
                        "has_actions": has_actions,
                    }
                )
        return result
