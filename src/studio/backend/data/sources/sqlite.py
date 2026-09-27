"""
SQLite metric source following the `marl.logging.sql_logger` schema (`init.sql`).

Databases (`*.sqlite`, `*.db`) are looked up in the run directory, then in the experiment directory.
An experiment-level database holds several runs, so its rows are filtered by the run's seed.
Only the `test` table (pivoted from `test` x `test_metric`) is exposed for now.
"""

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import orjson
import polars as pl

from ..cache import LRU, FileKey, file_key
from ..issues import Issue, Level, exception_detail
from .base import TIME_STEP, TIMESTAMP, TableInfo, latest_step_by_scan

SUFFIXES = (".sqlite", ".db")
TEST_TABLE = "test"

_probe_cache = LRU[tuple, TableInfo | Issue | None](4096)
_frame_cache = LRU[tuple, pl.DataFrame](256)
_dir_cache = LRU[tuple, list[Path]](4096)


def _connect(db: Path) -> sqlite3.Connection:
    """Read-only connection. @ai-generated"""
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def _run_seed(rundir: Path) -> int | None:
    """Seed of the run from `run.json`, else from the `run-<n>` directory name. @ai-generated"""
    try:
        seed = orjson.loads((rundir / "run.json").read_bytes()).get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    except (OSError, orjson.JSONDecodeError, AttributeError):
        pass
    match = re.fullmatch(r"run-(\d+)", rundir.name)
    return int(match.group(1)) if match else None


def _parse_location(location: str) -> tuple[Path, str, int | None]:
    """`<db>#<table>[@seed=<n>]` -> (db, table, seed). @ai-generated"""
    db, _, rest = location.partition("#")
    table, _, seed = rest.partition("@seed=")
    return Path(db), table, int(seed) if seed else None


def _epoch(value: object) -> float | None:
    """@ai-generated"""
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def _experiment_databases(directory: Path) -> list[Path]:
    """Databases of an experiment directory, cached by the directory's mtime. @ai-generated"""
    try:
        mtime = os.stat(directory).st_mtime_ns
    except OSError:
        return []

    def list_dbs() -> list[Path]:
        try:
            return sorted(Path(e.path) for e in os.scandir(directory) if e.name.endswith(SUFFIXES) and e.is_file())
        except OSError:
            return []

    return _dir_cache.get_or_compute((str(directory), mtime), list_dbs)


class SQLiteSource:
    name = "sqlite"

    def _databases(self, rundir: Path, files: dict[str, FileKey]) -> list[tuple[Path, int | None]]:
        """(db, seed filter) pairs: run-level databases first, then experiment-level ones. @ai-generated"""
        found: list[tuple[Path, int | None]] = [(rundir / name, None) for name in sorted(files) if name.endswith(SUFFIXES)]
        experiment_dbs = _experiment_databases(rundir.parent)
        if experiment_dbs:
            seed = _run_seed(rundir)
            found += [(db, seed) for db in experiment_dbs]
        return found

    def tables(self, rundir: Path, scope: str, files: dict[str, FileKey]) -> tuple[list[TableInfo], list[Issue]]:
        """@ai-generated"""
        tables, issues = list[TableInfo](), list[Issue]()
        for db, seed in self._databases(rundir, files):
            if any(t.name == TEST_TABLE for t in tables):
                break
            if db.parent != rundir and seed is None:
                continue
            key = (file_key(db), seed, scope)
            probed = _probe_cache.get_or_compute(key, lambda db=db, seed=seed: self._probe(db, seed, scope))
            if isinstance(probed, Issue):
                issues.append(probed)
            elif probed is not None:
                tables.append(probed)
        return tables, issues

    def _probe(self, db: Path, seed: int | None, scope: str) -> TableInfo | Issue | None:
        """@ai-generated"""
        try:
            with _connect(db) as con:
                names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"test", "test_metric"} <= names:
                    return None
                if seed is None:
                    rows = con.execute("SELECT DISTINCT key FROM test_metric").fetchall()
                else:
                    if "run" not in names:
                        return None
                    rows = con.execute(
                        "SELECT DISTINCT m.key FROM test_metric m JOIN test t ON m.test = t.id JOIN run r ON t.run = r.id WHERE r.seed = ?",
                        (seed,),
                    ).fetchall()
                    if not rows:
                        return None  # This run is not in the database
        except sqlite3.Error as exc:
            return Issue(
                Level.WARNING,
                "unreadable-table",
                f"Database {db.name} cannot be read.",
                f"table:{scope.removeprefix('run:')}/{TEST_TABLE}",
                path=db.name,
                detail=exception_detail(exc),
            )
        columns = {TIME_STEP: "num", TIMESTAMP: "num"} | {str(row[0]): "num" for row in rows}
        location = f"{db}#{TEST_TABLE}" + (f"@seed={seed}" if seed is not None else "")
        return TableInfo(TEST_TABLE, self.name, location, tuple(columns), kinds=columns)  # type: ignore[arg-type]

    def _read(self, location: str) -> pl.DataFrame:
        """Pivot `test` x `test_metric` into one row per test, in insertion order. @ai-generated"""
        db, _, seed = _parse_location(location)
        query = "SELECT t.id, t.time_step, t.timestamp, m.key, m.value FROM test t JOIN test_metric m ON m.test = t.id"
        params: tuple = ()
        if seed is not None:
            query += " JOIN run r ON t.run = r.id WHERE r.seed = ?"
            params = (seed,)
        with _connect(db) as con:
            rows = con.execute(query + " ORDER BY t.id", params).fetchall()
        schema = {"id": pl.Int64, "time_step": pl.Float64, "timestamp": pl.String, "key": pl.String, "value": pl.Float64}
        long = pl.DataFrame(rows, schema=schema, orient="row")
        if long.is_empty():
            return pl.DataFrame(schema={TIME_STEP: pl.Float64, TIMESTAMP: pl.Float64})
        wide = long.pivot(on="key", index=["id", "time_step", "timestamp"], values="value", aggregate_function="first").sort("id")
        stamps = [_epoch(v) for v in wide["timestamp"].to_list()]
        return wide.with_columns(pl.Series(TIMESTAMP, stamps, dtype=pl.Float64)).drop("id", "timestamp")

    def scan(self, table: TableInfo) -> pl.LazyFrame:
        """@ai-generated"""
        key = (table.location, file_key(table.file))
        return _frame_cache.get_or_compute(key, lambda: self._read(table.location)).lazy()

    def latest_step(self, table: TableInfo) -> int | None:
        """@ai-generated"""
        return latest_step_by_scan(self.scan(table))
