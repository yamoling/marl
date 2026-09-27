"""JSON Lines metric source (`*.jsonl`, one object per line), plus JSON array files (`*.json`)."""

from pathlib import Path

import orjson
import polars as pl

from ..cache import LRU, FileKey, file_key
from ..issues import Issue, Level, exception_detail
from .base import TIME_STEP, TableInfo, latest_step_by_scan, normalise, schema_of

SCHEMA_ROWS = 200
TAIL_BYTES = 4096
EXCLUDED_JSON = {"run.json", "actions.json", "experiment.json"}

_probe_cache = LRU[tuple, TableInfo | Issue | None](16_384)
_latest_cache = LRU[tuple, int | None](16_384)


def _is_json_array(path: Path) -> bool:
    """@ai-generated"""
    try:
        with open(path, "rb") as f:
            return f.read(64).lstrip().startswith(b"[")
    except OSError:
        return False


class JSONLSource:
    name = "jsonl"

    def tables(self, rundir: Path, scope: str, files: dict[str, FileKey]) -> tuple[list[TableInfo], list[Issue]]:
        """@ai-generated"""
        tables, issues = list[TableInfo](), list[Issue]()
        for name in sorted(files):
            if not name.endswith((".jsonl", ".json")):
                continue
            if name.endswith(".json") and (name in EXCLUDED_JSON or "action" in name):
                continue
            path = rundir / name
            probed = _probe_cache.get_or_compute((file_key(path), scope), lambda path=path: self._probe(path, scope))
            if isinstance(probed, Issue):
                issues.append(probed)
            elif probed is not None:
                tables.append(probed)
        return tables, issues

    def _read_sample(self, path: Path) -> pl.DataFrame:
        """@ai-generated"""
        if path.suffix == ".json":
            return pl.read_json(path).head(SCHEMA_ROWS)
        lines = []
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    lines.append(orjson.loads(line))
                if len(lines) >= SCHEMA_ROWS:
                    break
        return pl.DataFrame(lines, infer_schema_length=None)

    def _probe(self, path: Path, scope: str) -> TableInfo | Issue | None:
        """@ai-generated"""
        if path.suffix == ".json" and not _is_json_array(path):
            return None
        try:
            df = self._read_sample(path)
        except (pl.exceptions.PolarsError, OSError, ValueError, TypeError) as exc:
            return Issue(
                Level.WARNING,
                "unreadable-table",
                f"Table {path.name} cannot be read.",
                f"table:{scope.removeprefix('run:')}/{path.stem}",
                path=path.name,
                detail=exception_detail(exc),
            )
        columns = schema_of(df)
        return TableInfo(path.stem, self.name, str(path), tuple(columns), kinds=columns)

    def scan(self, table: TableInfo) -> pl.LazyFrame:
        """@ai-generated"""
        if not table.columns:
            return pl.LazyFrame()
        if table.location.endswith(".json"):
            lf = pl.read_json(table.location).lazy()
        else:
            lf = pl.scan_ndjson(table.location, infer_schema_length=None, ignore_errors=True)
        return normalise(lf, table.columns)

    def latest_step(self, table: TableInfo) -> int | None:
        """@ai-generated"""
        if TIME_STEP not in table.columns:
            return None
        return _latest_cache.get_or_compute((file_key(table.location),), lambda: self._latest_step(table))

    def _latest_step(self, table: TableInfo) -> int | None:
        """Tail read of the last complete line for JSONL files, full scan otherwise. @ai-generated"""
        if table.location.endswith(".jsonl"):
            try:
                with open(table.location, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - TAIL_BYTES))
                    lines = [line for line in f.read().split(b"\n") if line.strip()]
                for line in reversed(lines):
                    try:
                        value = orjson.loads(line).get(TIME_STEP)
                    except (orjson.JSONDecodeError, AttributeError):
                        continue  # Partial line
                    if isinstance(value, (int, float)):
                        return int(value)
            except OSError:
                return None
        return latest_step_by_scan(self.scan(table))
