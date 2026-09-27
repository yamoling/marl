"""Common definitions of the metric sources."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal, Protocol

import polars as pl

from ..issues import Issue

logger = logging.getLogger(__name__)

ColumnKind = Literal["num", "bool", "str"]

TIME_STEP = "time_step"
TIMESTAMP = "timestamp_sec"
EPISODE_NUM = "episode_num"
X_COLUMNS = (TIME_STEP, TIMESTAMP, EPISODE_NUM)


@dataclass(frozen=True)
class TableInfo:
    name: str
    """File stem, e.g. `test` or `test-policy-on-test-envs`."""
    source: str
    """`csv`, `jsonl` or `sqlite`."""
    location: str
    """File path, or `<db path>#<table>` for SQLite."""
    names: tuple[str, ...] = ()
    """Column names in file order (cheap: read from the header)."""
    kinds: dict[str, ColumnKind] | None = field(default=None, hash=False, compare=False)
    """Column kinds when known eagerly; otherwise computed on demand by `loader`."""
    loader: Callable[[], dict[str, ColumnKind]] | None = field(default=None, hash=False, compare=False, repr=False)
    key: tuple[str, int, int] | None = field(default=None, hash=False, compare=False, repr=False)
    """`file_key` of the file when the table was probed."""

    @cached_property
    def columns(self) -> dict[str, ColumnKind]:
        """
        Column kinds, computed lazily (and cached per file by the source): listing experiments only
        needs the names, while catalogs and series need the kinds.

        @ai-generated
        """
        if self.kinds is not None:
            return self.kinds
        if self.loader is None:
            return {}
        try:
            return self.loader()
        except Exception:
            logger.exception("Could not read the column kinds of %s", self.location)
            return {}

    @property
    def x_columns(self) -> tuple[str, ...]:
        """@ai-generated"""
        return x_columns_of(self.names)

    @property
    def file(self) -> Path:
        """The file backing the table (without the `#table` suffix of SQLite locations). @ai-generated"""
        return Path(self.location.split("#", 1)[0])

    @property
    def plottable(self) -> list[str]:
        """Numeric and boolean columns that are not x columns, in file order. @ai-generated"""
        return [c for c, kind in self.columns.items() if kind != "str" and c not in X_COLUMNS]


class MetricSource(Protocol):
    name: str

    def tables(self, rundir: Path, scope: str, files: dict[str, tuple[str, int, int] | None]) -> tuple[list[TableInfo], list[Issue]]:
        """
        Probe the tables of `rundir`, whose regular files are `files` (name -> `file_key`, when
        known). `scope` is the run scope used for issues (`run:<run-id>`).
        """
        ...

    def scan(self, table: TableInfo) -> pl.LazyFrame:
        """Lazy frame where num/bool columns are Float64 (bool as 0/1)."""
        ...

    def latest_step(self, table: TableInfo) -> int | None:
        """Largest `time_step` of the table, or None when it has no row."""
        ...


def kind_of(dtype: pl.DataType, all_null: bool) -> ColumnKind:
    """
    Classify a column. Columns that are entirely null in the sample (e.g. a metric appearing
    mid-file) are considered numeric.

    @ai-generated
    """
    if dtype == pl.Boolean:
        return "bool"
    if dtype.is_numeric():
        return "num"
    if dtype == pl.Null or all_null:
        return "num"
    return "str"


def schema_of(df: pl.DataFrame) -> dict[str, ColumnKind]:
    """@ai-generated"""
    schema = df.schema
    nulls = df.null_count().row(0) if df.width > 0 else ()
    return {name: kind_of(dtype, n == df.height) for (name, dtype), n in zip(schema.items(), nulls)}


def x_columns_of(columns: dict[str, ColumnKind] | tuple[str, ...]) -> tuple[str, ...]:
    """@ai-generated"""
    return tuple(c for c in X_COLUMNS if c in columns)


def normalise(lf: pl.LazyFrame, columns: dict[str, ColumnKind]) -> pl.LazyFrame:
    """
    Cast num/bool columns to Float64 (non-strict), so that bools become 0/1 and unparsable values null.

    @ai-generated
    """
    exprs = []
    for name, kind in columns.items():
        if kind == "str":
            continue
        exprs.append(pl.col(name).cast(pl.Float64, strict=False))
    return lf.with_columns(exprs) if exprs else lf


def latest_step_by_scan(lf: pl.LazyFrame) -> int | None:
    """@ai-generated"""
    try:
        value = lf.select(pl.col(TIME_STEP).cast(pl.Float64, strict=False).max()).collect().item()
    except (pl.exceptions.PolarsError, OSError, ValueError):
        return None
    if value is None or math.isnan(value):
        return None
    return int(value)
