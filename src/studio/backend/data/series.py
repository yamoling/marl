"""
Series computation: per-run bucketing, cross-run statistics and M4 downsampling. No smoothing.

The semantics match the legacy `marl.utils.stats.compute_experiment_results` (round x to the
closest multiple of the resolution, mean within each (run, bucket), then statistics across runs),
except that nulls are dropped per metric rather than per row, and wall time is relative to each
run's own start.
"""

import logging
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Literal, get_args

import numpy as np
import polars as pl

from .cache import LRU, ByteLRU, file_key
from .issues import Issue, Level, exception_detail
from .records import ExperimentRecord, RunRecord
from .sources import TableInfo, read_numeric
from .sources.base import TIME_STEP, TIMESTAMP

logger = logging.getLogger(__name__)

XMode = Literal["time_step", "wall_time"]
Center = Literal["mean", "median", "none"]
Band = Literal["ci95", "std", "minmax", "none"]

MAX_POINTS_LIMIT = 5000
EXACT_MAX_DISTINCT = 2000
AUTO_BUCKETS = 500
X_SOURCE = {"time_step": TIME_STEP, "wall_time": TIMESTAMP}

COLUMN_CACHE_BYTES = 1 << 30
_result_cache = LRU[tuple, "SeriesResult"](256)
_read_pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="studio-series")
"""Per-run table reads run in parallel: Polars releases the GIL while parsing."""


@dataclass(frozen=True)
class SeriesQuery:
    experiment: str
    table: str
    metric: str
    x: XMode = "time_step"
    runs: tuple[str, ...] | None = None
    """None means all runs."""
    center: Center = "mean"
    band: Band = "ci95"
    resolution: int | None = None
    """Bucket size on the x axis; None means automatic."""
    include_runs: bool = False
    max_points: int = 1000

    @classmethod
    def from_json(cls, data: Any) -> "SeriesQuery":
        """
        Validate an API query (snake_case keys, defaults as in api-contract.md).
        Raises TypeError for values of the wrong type and ValueError for invalid values.

        @ai-generated
        """
        if not isinstance(data, dict):
            raise TypeError("A query must be an object")
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"Unknown query fields: {', '.join(sorted(unknown))}")
        for key in ("experiment", "table", "metric"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"`{key}` must be a non-empty string")
        choices = {"x": get_args(XMode), "center": get_args(Center), "band": get_args(Band)}
        for key, allowed in choices.items():
            if key in data and data[key] not in allowed:
                raise ValueError(f"`{key}` must be one of {', '.join(allowed)}")
        runs = data.get("runs")
        if runs is not None and (not isinstance(runs, list) or not all(isinstance(r, str) for r in runs)):
            raise ValueError("`runs` must be null or a list of run ids")
        resolution = data.get("resolution")
        if resolution is not None and (isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1):
            raise ValueError("`resolution` must be null or a positive integer")
        max_points = data.get("max_points", 1000)
        if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
            raise ValueError("`max_points` must be a positive integer")
        include_runs = data.get("include_runs", False)
        if not isinstance(include_runs, bool):
            raise TypeError("`include_runs` must be a boolean")
        return cls(
            experiment=data["experiment"],
            table=data["table"],
            metric=data["metric"],
            x=data.get("x", "time_step"),
            runs=tuple(runs) if runs is not None else None,
            center=data.get("center", "mean"),
            band=data.get("band", "ci95"),
            resolution=resolution,
            include_runs=include_runs,
            max_points=max_points,
        ).normalised()

    def normalised(self) -> "SeriesQuery":
        """Clamp `max_points` and force `include_runs` when there is no centre line. @ai-generated"""
        return replace(
            self,
            max_points=max(4, min(self.max_points, MAX_POINTS_LIMIT)),
            include_runs=self.include_runs or self.center == "none",
        )


def _clean(values: Sequence[float] | np.ndarray) -> list[float | None]:
    """@ai-generated"""
    return [float(v) if v is not None and math.isfinite(v) else None for v in values]


@dataclass
class RunSeries:
    run: str
    seed: int | None
    x: list[float]
    y: list[float | None]

    def to_json(self) -> dict[str, Any]:
        """@ai-generated"""
        return {"run": self.run, "seed": self.seed, "x": [float(v) for v in self.x], "y": _clean(self.y)}  # type: ignore[arg-type]


@dataclass
class SeriesResult:
    x: list[float]
    center: list[float | None] | None
    lo: list[float | None] | None
    hi: list[float | None] | None
    n: list[int]
    runs: list[RunSeries]
    used_runs: list[str]
    missing_runs: list[str]
    resolution: int
    issues: list[Issue] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The API's `SeriesResult` (NaN/inf become null). @ai-generated"""
        return {
            "x": [float(v) for v in self.x],
            "center": _clean(self.center) if self.center is not None else None,  # type: ignore[arg-type]
            "lo": _clean(self.lo) if self.lo is not None else None,  # type: ignore[arg-type]
            "hi": _clean(self.hi) if self.hi is not None else None,  # type: ignore[arg-type]
            "n": [int(v) for v in self.n],
            "runs": [r.to_json() for r in self.runs],
            "used_runs": self.used_runs,
            "missing_runs": self.missing_runs,
            "resolution": self.resolution,
            "issues": [i.to_json() for i in self.issues],
        }


# ---------------------------------------------------------------- Per-run data


def _usable(table: TableInfo | None, metric: str, x: XMode) -> bool:
    """@ai-generated"""
    return table is not None and table.columns.get(metric) in ("num", "bool") and X_SOURCE[x] in table.columns


def _columns_nbytes(columns: dict[str, np.ndarray]) -> int:
    return sum(a.nbytes for a in columns.values())


_columns_cache = ByteLRU[tuple, dict[str, np.ndarray]](COLUMN_CACHE_BYTES, _columns_nbytes)
"""Numeric columns of each table file, shared by all the metrics of the table."""


def table_columns(table: TableInfo) -> dict[str, np.ndarray]:
    """
    Every numeric/bool column of a table as a float64 array (nulls as NaN), parsed once per file
    version: cached by (location, mtime, size) in an LRU bounded by bytes.

    @ai-generated
    """
    key = (table.location, file_key(table.file))
    return _columns_cache.get_or_compute(key, lambda: _read_columns(table))


def _read_columns(table: TableInfo) -> dict[str, np.ndarray]:
    """@ai-generated"""
    names = [c for c, kind in table.columns.items() if kind != "str"]
    if not names:
        return {}
    df = read_numeric(table, names)
    return {c: df[c].cast(pl.Float64, strict=False).to_numpy() for c in names}


def load_run(table: TableInfo, metric: str, x: XMode) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite (x, y) values of one run, in file order. Wall time is relative to the run's first
    timestamp (over all rows of the table).

    @ai-generated
    """
    columns = table_columns(table)
    xs, ys = columns.get(X_SOURCE[x]), columns.get(metric)
    if xs is None or ys is None or len(xs) == 0:
        return np.empty(0), np.empty(0)
    if x == "wall_time":
        finite = xs[np.isfinite(xs)]
        xs = xs - finite.min() if len(finite) else xs
    mask = np.isfinite(xs) & np.isfinite(ys)
    return xs[mask], ys[mask]


# ---------------------------------------------------------------- Resolution and bucketing


def nice_number(value: float) -> int:
    """The number of the form {1, 2, 5} x 10^k closest to `value` (log scale), at least 1. @ai-generated"""
    if value <= 1:
        return 1
    exponent = math.floor(math.log10(value))
    candidates = [m * 10**e for e in (exponent - 1, exponent, exponent + 1) for m in (1, 2, 5)]
    best = min(candidates, key=lambda c: abs(math.log(c) - math.log(value)))
    return max(1, int(best))


def auto_resolution(xs: Sequence[np.ndarray]) -> int:
    """
    Exact x values (resolution = gcd of the x values) when all runs lie on an integer lattice with
    fewer than 2000 distinct values; otherwise a nice number giving about 500 buckets.

    @ai-generated
    """
    values = [x for x in xs if len(x) > 0]
    if not values:
        return 1
    span = float(max(x.max() for x in values) - min(x.min() for x in values))
    distinct_per_run = list[np.ndarray]()
    for x in values:
        distinct = _distinct(x)
        if distinct is None:
            return nice_number(span / AUTO_BUCKETS)
        distinct_per_run.append(distinct)
    distinct = np.unique(np.concatenate(distinct_per_run))
    if len(distinct) < EXACT_MAX_DISTINCT and np.all(distinct == np.round(distinct)):
        gcd = int(np.gcd.reduce(np.abs(distinct.astype(np.int64))))
        return max(gcd, 1)
    return nice_number(span / AUTO_BUCKETS)


def _distinct(x: np.ndarray) -> np.ndarray | None:
    """
    Distinct values of `x`, or None as soon as there are at least `EXACT_MAX_DISTINCT` of them.
    O(n) for sorted arrays (x columns are time steps or timestamps, almost always sorted).

    @ai-generated
    """
    if len(x) > 1 and np.all(x[1:] >= x[:-1]):
        changes = np.flatnonzero(x[1:] != x[:-1])
        if len(changes) + 1 >= EXACT_MAX_DISTINCT:
            return None
        return np.concatenate([x[:1], x[changes + 1]])
    distinct = np.unique(x)
    return None if len(distinct) >= EXACT_MAX_DISTINCT else distinct


def bucket(x: np.ndarray, y: np.ndarray, resolution: int) -> pl.DataFrame:
    """Round x to the closest multiple of `resolution` (half to even, as Polars) and average y per bucket. @ai-generated"""
    return (
        pl.DataFrame({"x": x, "y": y})
        .with_columns(xb=(pl.col("x") / resolution).round(0) * resolution)
        .group_by("xb")
        .agg(pl.col("y").mean())
        .sort("xb")
    )


def bucket_runs(runs: Sequence[tuple[np.ndarray, np.ndarray]], resolution: int) -> pl.DataFrame:
    """
    `bucket` for several runs at once: one frame `(run, xb, y)` with the mean of y per (run, bucket),
    computed by a single group-by over the concatenated runs.

    @ai-generated
    """
    frames = []
    for i, (x, y) in enumerate(runs):
        xb, mean = _bucket_sorted(x, y, resolution)
        if xb is None:  # Unsorted x: generic group-by
            df = bucket(x, y, resolution)
            xb, mean = df["xb"].to_numpy(), df["y"].to_numpy()
        frames.append(pl.DataFrame({"run": np.full(len(xb), i, dtype=np.int32), "xb": xb, "y": mean}))
    return pl.concat(frames)


def _bucket_sorted(x: np.ndarray, y: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    O(n) bucketing of a run whose x is non-decreasing (the usual case): contiguous segments of equal
    buckets are averaged with `np.add.reduceat`. Rounding is half to even, like Polars' `round`.
    Returns (None, None) if x is not sorted.

    @ai-generated
    """
    xb = np.round(x / resolution) * resolution
    if len(xb) > 1 and not np.all(xb[1:] >= xb[:-1]):
        return None, None
    starts = np.concatenate([[0], np.flatnonzero(xb[1:] != xb[:-1]) + 1])
    counts = np.diff(np.append(starts, len(xb)))
    return xb[starts], np.add.reduceat(y, starts) / counts


def m4_indices(y: np.ndarray, max_points: int) -> np.ndarray:
    """
    M4 selection: split the indices into `max_points // 4` groups and keep the first, min, max and last
    index of each group. Keeps the global min and max. Returns all indices if already small enough.

    @ai-generated
    """
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    groups = max(1, max_points // 4)
    keep = set[int]()
    finite = np.where(np.isfinite(y), y, np.nan)
    for chunk in np.array_split(np.arange(n), groups):
        if len(chunk) == 0:
            continue
        keep.update((int(chunk[0]), int(chunk[-1])))
        values = finite[chunk]
        if np.all(np.isnan(values)):
            continue
        keep.update((int(chunk[np.nanargmin(values)]), int(chunk[np.nanargmax(values)])))
    return np.array(sorted(keep), dtype=np.int64)


# ---------------------------------------------------------------- Computation


def _issue(level: Level, code: str, message: str, detail: str | None = None) -> Issue:
    return Issue(level, code, message, "experiment", detail=detail)


def _involved_files(record: ExperimentRecord, runs: list[RunRecord], table: str) -> tuple:
    """@ai-generated"""
    return tuple(file_key(run.tables[table].file) if table in run.tables else None for run in runs)


def compute(record: ExperimentRecord, query: SeriesQuery) -> SeriesResult:
    """
    Compute one series of `record` (see backend.md §6). Results are cached by the query and the
    stats of the involved files.

    @ai-generated
    """
    query = query.normalised()
    issues = list[Issue]()
    if query.runs is None:
        runs = list(record.runs)
    else:
        runs = [r for r in (record.run(run_id) for run_id in query.runs) if r is not None]
        unknown = [run_id for run_id in query.runs if record.run(run_id) is None]
        if unknown:
            issues.append(_issue(Level.WARNING, "unknown-run", f"Unknown runs: {', '.join(unknown)}."))
    key = (query, record.id, tuple(r.id for r in runs), _involved_files(record, runs, query.table))
    cached = _result_cache.get(key)
    if cached is not None and not issues:
        return cached
    result = _compute(record, query, runs, issues)
    _result_cache.put(key, result)
    return result


def _compute(record: ExperimentRecord, query: SeriesQuery, runs: list[RunRecord], issues: list[Issue]) -> SeriesResult:
    """@ai-generated"""
    missing = [run_id for run_id in (query.runs or ()) if record.run(run_id) is None]

    def load(run: RunRecord) -> tuple[np.ndarray, np.ndarray] | None:
        table = run.tables.get(query.table)
        if table is None or not _usable(table, query.metric, query.x):
            return None
        return load_run(table, query.metric, query.x)

    data = list[tuple[RunRecord, np.ndarray, np.ndarray]]()
    for run, loaded in zip(runs, _read_pool.map(load, runs)):
        if loaded is None or len(loaded[0]) == 0:
            missing.append(run.id)
        else:
            data.append((run, *loaded))
    resolution = query.resolution or auto_resolution([x for _, x, _ in data])
    if not data:
        issues.append(_issue(Level.INFO, "missing-metric", f"No selected run has {query.table}/{query.metric}."))
        return SeriesResult([], None, None, None, [], [], [], missing, resolution, issues)

    stacked = bucket_runs([(x, y) for _, x, y in data], resolution)
    stats = (
        stacked.group_by("xb")
        .agg(
            mean=pl.col("y").mean(),
            median=pl.col("y").median(),
            std=pl.col("y").std(ddof=1).fill_null(0.0),
            min=pl.col("y").min(),
            max=pl.col("y").max(),
            n=pl.len(),
        )
        .sort("xb")
    )
    xs = stats["xb"].to_numpy()
    n = stats["n"].to_numpy()
    mean = stats["mean"].to_numpy()
    centre = stats["median"].to_numpy() if query.center == "median" else mean
    std = stats["std"].to_numpy()
    lo = hi = None
    if query.center != "none":
        match query.band:
            case "ci95":
                half = 1.96 * std / np.sqrt(n)
                lo, hi = centre - half, centre + half
            case "std":
                lo, hi = centre - std, centre + std
            case "minmax":
                lo, hi = stats["min"].to_numpy(), stats["max"].to_numpy()
            case "none":
                pass
    keep = m4_indices(centre, query.max_points)
    run_series = list[RunSeries]()
    if query.include_runs:
        per_run = stacked.sort("run", "xb").partition_by("run", as_dict=True, maintain_order=True)
        for i, (run, _, _) in enumerate(data):
            df = per_run[(i,)]
            rx, ry = df["xb"].to_numpy(), df["y"].to_numpy()
            idx = m4_indices(ry, query.max_points)
            run_series.append(RunSeries(run.id, run.seed, rx[idx].tolist(), ry[idx].tolist()))
    return SeriesResult(
        x=xs[keep].tolist(),
        center=centre[keep].tolist() if query.center != "none" else None,
        lo=lo[keep].tolist() if lo is not None else None,
        hi=hi[keep].tolist() if hi is not None else None,
        n=n[keep].astype(int).tolist(),
        runs=run_series,
        used_runs=[run.id for run, _, _ in data],
        missing_runs=missing,
        resolution=resolution,
        issues=issues,
    )


BatchItem = SeriesResult | Issue


def compute_batch(get_record: Callable[[str], ExperimentRecord | None], queries: Sequence[SeriesQuery | dict]) -> list[BatchItem]:
    """
    Compute each query independently: an invalid query or a failure becomes an `Issue` for that
    query only (logged with its traceback).

    @ai-generated
    """
    results = list[BatchItem]()
    for raw in queries:
        try:
            query = raw if isinstance(raw, SeriesQuery) else SeriesQuery.from_json(raw)
        except (ValueError, TypeError) as exc:
            results.append(_issue(Level.ERROR, "invalid-query", str(exc)))
            continue
        try:
            record = get_record(query.experiment)
            if record is None:
                results.append(_issue(Level.ERROR, "unknown-experiment", f"Unknown experiment {query.experiment}."))
                continue
            results.append(compute(record, query))
        except Exception as exc:
            logger.exception("Series query failed: %s", query)
            results.append(
                _issue(
                    Level.ERROR, "series-failed", f"The series {query.table}/{query.metric} could not be computed.", exception_detail(exc)
                )
            )
    return results


def batch_to_json(items: Sequence[BatchItem]) -> list[dict[str, Any]]:
    """`[{ok: true, result} | {ok: false, issue}]`. @ai-generated"""
    return [{"ok": True, "result": i.to_json()} if isinstance(i, SeriesResult) else {"ok": False, "issue": i.to_json()} for i in items]
