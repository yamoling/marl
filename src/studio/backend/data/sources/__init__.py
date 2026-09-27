"""Registry of metric sources and table discovery for a run directory."""

import os
from pathlib import Path

import polars as pl

from ..cache import FileKey, file_key
from ..issues import Issue, Level
from .base import X_COLUMNS, MetricSource, TableInfo
from .csv import CSVSource
from .jsonl import JSONLSource
from .sqlite import SQLiteSource

SOURCES: dict[str, MetricSource] = {s.name: s for s in (CSVSource(), JSONLSource(), SQLiteSource())}
"""Sources in priority order: on a name collision, the first source wins."""


def list_files(rundir: Path) -> dict[str, FileKey]:
    """Regular files of a directory: name -> `file_key` (one `scandir`). @ai-generated"""
    try:
        with os.scandir(rundir) as it:
            return {e.name: file_key(e.path) for e in it if e.is_file()}
    except OSError:
        return {}


def discover_tables(rundir: Path, scope: str, files: dict[str, FileKey] | None = None) -> tuple[dict[str, TableInfo], list[Issue]]:
    """
    Probe every source for the tables of `rundir` (whose file names may be given, to avoid listing
    it again). On a name collision, the table of the higher-priority source (CSV first) is kept and
    an `info` issue is recorded.

    @ai-generated
    """
    if files is None:
        files = list_files(rundir)
    tables = dict[str, TableInfo]()
    issues = list[Issue]()
    for source in SOURCES.values():
        found, source_issues = source.tables(rundir, scope, files)
        issues.extend(source_issues)
        for table in found:
            if table.name in tables:
                kept = tables[table.name]
                issues.append(
                    Issue(
                        Level.INFO,
                        "duplicate-table",
                        f"Table {table.name} exists in {kept.source} and {table.source}; the {kept.source} one is used.",
                        f"table:{scope.removeprefix('run:')}/{table.name}",
                        path=table.file.name,
                    )
                )
                continue
            tables[table.name] = table
    return tables, issues


def scan(table: TableInfo) -> pl.LazyFrame:
    """@ai-generated"""
    return SOURCES[table.source].scan(table)


def read_numeric(table: TableInfo, names: list[str]) -> pl.DataFrame:
    """
    The given num/bool columns of a table, as Float64, read eagerly. Sources may provide a faster
    path than `scan(...).collect()` (`read_numeric` method).

    @ai-generated
    """
    source = SOURCES[table.source]
    reader = getattr(source, "read_numeric", None)
    if reader is not None:
        return reader(table, names)
    return source.scan(table).select(pl.col(c).cast(pl.Float64, strict=False) for c in names).collect()


def latest_step(table: TableInfo) -> int | None:
    """@ai-generated"""
    return SOURCES[table.source].latest_step(table)


__all__ = ["SOURCES", "X_COLUMNS", "MetricSource", "TableInfo", "discover_tables", "latest_step", "read_numeric", "scan"]
