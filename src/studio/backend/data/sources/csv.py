"""
CSV metric source: every `*.csv` file of a run directory is a table.

Listing experiments only opens each file once, to read its header line and its last line (for
`latest_step`). Column kinds (which need a sample of rows) are computed lazily, and the file is
parsed with an explicit schema when a series needs it.
"""

import csv
import io
import os
from pathlib import Path

import polars as pl

from ..cache import LRU, FileKey, file_key
from ..issues import Issue, Level, exception_detail
from .base import TIME_STEP, ColumnKind, TableInfo, latest_step_by_scan, normalise, schema_of

SCHEMA_ROWS = 200
TAIL_BYTES = 4096
HEADER_BYTES = 1 << 16
READ_THREADS = 2
"""Polars threads per file read (files are read in parallel by the series computation)."""

_probe_cache = LRU[tuple, "TableInfo | Issue"](32_768)
_kinds_cache = LRU[FileKey, dict[str, ColumnKind]](32_768)
_latest_cache = LRU[FileKey, int | None](32_768)


def _polars_types(names: tuple[str, ...], kinds: dict[str, ColumnKind]) -> dict[str, pl.DataType]:
    """@ai-generated"""
    types: dict[str, pl.DataType] = {}
    for name in names:
        kind = kinds.get(name, "str")
        types[name] = pl.Float64() if kind == "num" else pl.Boolean() if kind == "bool" else pl.String()
    return types


def _parse_header(line: bytes) -> tuple[str, ...]:
    """@ai-generated"""
    text = line.decode(errors="replace").rstrip("\r\n")
    return tuple(next(csv.reader([text]), [])) if text else ()


def _last_value(chunk: bytes, index: int, partial_first: bool) -> int | None:
    """
    Value of column `index` in the last complete line of `chunk` (the tail of a file). None when
    there is no data line. Raises ValueError if the last line cannot be parsed.

    @ai-generated
    """
    if not chunk.strip():
        return None
    lines = chunk.split(b"\n")
    if not chunk.endswith(b"\n"):
        lines = lines[:-1]  # The last line is being written
    if partial_first:
        lines = lines[1:]  # The first line of the chunk may be partial
    lines = [line for line in lines if line.strip()]
    if not lines:
        if partial_first:
            raise ValueError("No complete line in the tail of the file")
        return None
    row = next(csv.reader(io.StringIO(lines[-1].decode())))
    return int(float(row[index]))


class CSVSource:
    name = "csv"

    def tables(self, rundir: Path, scope: str, files: dict[str, FileKey]) -> tuple[list[TableInfo], list[Issue]]:
        """@ai-generated"""
        tables, issues = list[TableInfo](), list[Issue]()
        directory = str(rundir)
        for name in sorted(f for f in files if f.endswith(".csv")):
            path = os.path.join(directory, name)
            probed = self.probe(path, scope, files[name] or file_key(path))
            if isinstance(probed, Issue):
                issues.append(probed)
            else:
                tables.append(probed)
        return tables, issues

    def probe(self, path: Path | str, scope: str, key: FileKey = None) -> TableInfo | Issue:
        """Header (and tail, for `latest_step`) of a CSV file, cached by (path, mtime, size). @ai-generated"""
        if key is None:
            key = file_key(path)
        return _probe_cache.get_or_compute((key, scope), lambda: self._probe(Path(path), key, scope))

    def _probe(self, path: Path, key: FileKey, scope: str) -> TableInfo | Issue:
        """
        One open and at most two `pread`s: the header line, then the tail when there is a `time_step`
        column (the tail is also cached for `latest_step`).

        @ai-generated
        """
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                head = os.pread(fd, TAIL_BYTES, 0)
                end = head.find(b"\n")
                if end < 0 and len(head) == TAIL_BYTES:  # Very long header
                    head = os.pread(fd, HEADER_BYTES, 0)
                    end = head.find(b"\n")
                header_line = head if end < 0 else head[: end + 1]
                names = _parse_header(header_line)
                if TIME_STEP in names:
                    size = key[2] if key is not None else os.fstat(fd).st_size
                    start = max(len(header_line), size - TAIL_BYTES)
                    chunk = head[start:] if size <= len(head) else os.pread(fd, TAIL_BYTES, start)
                    try:
                        _latest_cache.put(key, _last_value(chunk, names.index(TIME_STEP), start > len(header_line)))
                    except (ValueError, IndexError, csv.Error, UnicodeDecodeError):
                        pass  # `latest_step` falls back to a scan
            finally:
                os.close(fd)
        except OSError as exc:
            return Issue(
                Level.WARNING,
                "unreadable-table",
                f"Table {path.name} cannot be read.",
                f"table:{scope.removeprefix('run:')}/{path.stem}",
                path=path.name,
                detail=exception_detail(exc),
            )
        if not names:
            return TableInfo(path.stem, self.name, str(path), (), kinds={}, key=key)
        return TableInfo(path.stem, self.name, str(path), names, loader=lambda: self.kinds(path, key), key=key)

    def kinds(self, path: Path, key: FileKey) -> dict[str, ColumnKind]:
        """Column kinds from the first rows, cached by the file stats at probe time. @ai-generated"""
        return _kinds_cache.get_or_compute(key, lambda: self._kinds(path))

    def _kinds(self, path: Path) -> dict[str, ColumnKind]:
        """@ai-generated"""
        try:
            df = pl.read_csv(
                io.BytesIO(_head(path, SCHEMA_ROWS)),
                infer_schema_length=SCHEMA_ROWS,
                truncate_ragged_lines=True,
                ignore_errors=True,
                n_threads=1,
            )
        except (pl.exceptions.PolarsError, OSError, UnicodeDecodeError):
            return {}
        return schema_of(df)

    def scan(self, table: TableInfo) -> pl.LazyFrame:
        """
        Lazy frame with an explicit schema (no inference): num/bool columns as Float64 (bools as 0/1).

        @ai-generated
        """
        kinds = table.columns
        if not kinds:
            return pl.LazyFrame()
        types = _polars_types(table.names, kinds)
        if len(set(table.names)) == len(table.names) and set(kinds) == set(table.names):
            lf = pl.scan_csv(table.location, schema=types, ignore_errors=True, truncate_ragged_lines=True)
        else:  # Duplicate header names: let Polars deduplicate them
            overrides = {k: v for k, v in types.items() if k in kinds}
            lf = pl.scan_csv(
                table.location, ignore_errors=True, infer_schema_length=10_000, schema_overrides=overrides, truncate_ragged_lines=True
            )
        return normalise(lf, kinds)

    def read_numeric(self, table: TableInfo, names: list[str]) -> pl.DataFrame:
        """
        Eager read of some num/bool columns with an explicit schema (no inference) and few threads, as
        several files are read in parallel. Bools become 0/1 Float64.

        @ai-generated
        """
        kinds = table.columns
        if len(set(table.names)) != len(table.names) or set(kinds) != set(table.names):
            return self.scan(table).select(pl.col(c).cast(pl.Float64, strict=False) for c in names).collect()
        df = pl.read_csv(
            table.location,
            schema=_polars_types(table.names, kinds),
            columns=names,
            ignore_errors=True,
            truncate_ragged_lines=True,
            n_threads=READ_THREADS,
        )
        return df.with_columns(pl.col(c).cast(pl.Float64, strict=False) for c in names)

    def latest_step(self, table: TableInfo) -> int | None:
        """Largest time step, from the last line of the file (tail read during the probe). @ai-generated"""
        if TIME_STEP not in table.names:
            return None
        key = table.key or file_key(table.location)
        return _latest_cache.get_or_compute(key, lambda: self._latest_step(table))

    def _latest_step(self, table: TableInfo) -> int | None:
        """@ai-generated"""
        try:
            value = tail_value(Path(table.location), TIME_STEP)
        except (OSError, ValueError, IndexError, csv.Error, UnicodeDecodeError):
            value = None
        if value is not None:
            return value
        return latest_step_by_scan(self.scan(table))


def _head(path: Path, n_rows: int) -> bytes:
    """The header and the first `n_rows` complete lines of a file. @ai-generated"""
    lines = list[bytes]()
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n") and lines:
                break  # Partial line being written
            lines.append(line)
            if len(lines) > n_rows:
                break
    return b"".join(lines)


def tail_value(path: Path, column: str) -> int | None:
    """
    Value of `column` in the last complete line of a CSV file, without reading the whole file.
    Returns None if the file has no data row. Raises ValueError if the last line cannot be parsed.

    @ai-generated
    """
    with open(path, "rb") as f:
        header_line = f.readline(HEADER_BYTES)
        header = _parse_header(header_line)
        if column not in header:
            return None
        size = os.fstat(f.fileno()).st_size
        start = max(len(header_line), size - TAIL_BYTES)
        f.seek(start)
        chunk = f.read()
    return _last_value(chunk, header.index(column), start > len(header_line))
