"""Performance caches and fast paths: lazy column kinds, column cache, listing quick key, O(n) bucketing."""

import json
import os
import time

import numpy as np
import pytest

from studio.backend.data import series as series_module
from studio.backend.data.cache import ByteLRU
from studio.backend.data.library import Library
from studio.backend.data.series import SeriesQuery, _bucket_sorted, _distinct, auto_resolution, bucket, bucket_runs, compute
from studio.backend.data.sources.csv import CSVSource


def q(experiment="healthy", table="test", metric="score-0", **kwargs) -> SeriesQuery:
    return SeriesQuery(experiment, table, metric, **kwargs)


def test_listing_does_not_parse_column_kinds(make_logs, monkeypatch):
    calls = []
    original = CSVSource._kinds
    monkeypatch.setattr(CSVSource, "_kinds", lambda self, path: calls.append(path) or original(self, path))
    library = Library(make_logs)
    summaries = library.list_summaries()
    assert len(summaries) == 11 and calls == []
    healthy = library.get("healthy")
    assert healthy is not None and healthy.runs[0].latest_step == 10_000  # tail read without kinds
    assert calls == []
    assert library.catalog("healthy")["default_metric"] == {"table": "test", "metric": "score-0"}
    assert len(calls) == 9  # 3 runs x 3 tables, once each
    library.catalog("healthy")
    assert len(calls) == 9


def test_probe_latest_step_edge_cases(tmp_path):
    source = CSVSource()
    small = tmp_path / "small.csv"
    small.write_text("a,time_step\n1,10\n2,20\n")
    big = tmp_path / "big.csv"
    big.write_text("a,time_step\n" + "".join(f"{i},{i * 10}\n" for i in range(5000)) + "7,99")  # partial last line
    long_header = tmp_path / "long.csv"
    names = [f"metric-{i:05d}" for i in range(1000)]
    long_header.write_text(",".join([*names, "time_step"]) + "\n" + ",".join(["0"] * 1000 + ["42"]) + "\n")
    header_only = tmp_path / "header.csv"
    header_only.write_text("a,time_step\n")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    expected = {small: 20, big: 49_990, long_header: 42, header_only: None, empty: None}
    for path, latest in expected.items():
        table = source.probe(path, "run:x")
        assert not (table is None)
        assert source.latest_step(table) == latest, path  # type: ignore[arg-type]
    table = source.probe(long_header, "run:x")
    assert table.names[-1] == "time_step" and len(table.names) == 1001  # type: ignore[union-attr]


def test_column_cache_is_shared_by_metrics_and_invalidated_on_change(make_logs, monkeypatch):
    reads = []
    original = series_module.read_numeric
    monkeypatch.setattr(series_module, "read_numeric", lambda table, names: reads.append(table.location) or original(table, names))
    series_module._columns_cache.clear()
    library = Library(make_logs)
    record = library.get("healthy")
    compute(record, q(metric="score-0"))
    compute(record, q(metric="exit_rate"))
    compute(record, q(metric="gems_collected", x="wall_time"))
    assert len(reads) == 3  # one parse per run file for three metrics
    with open(make_logs / "healthy" / "run-0" / "test.csv", "a") as f:
        f.write("0,0.5,10,100.0,1700000500.0,11000\n")
    result = compute(library.get("healthy"), q(metric="score-0"))
    assert len(reads) == 4 and result.x[-1] == 11_000  # only the modified file is parsed again


def test_byte_lru_is_bounded_by_bytes():
    cache = ByteLRU[str, np.ndarray](max_bytes=1000, sizeof=lambda a: a.nbytes)
    for i in range(5):
        cache.put(str(i), np.zeros(50))  # 400 bytes each
    assert cache.nbytes <= 1000 and len(cache) == 2 and cache.get("4") is not None and cache.get("0") is None
    cache.put("huge", np.zeros(1000))  # larger than the bound: not stored
    assert cache.get("huge") is None and cache.nbytes <= 1000
    cache.discard_if(lambda k: k == "4")
    assert cache.nbytes == 400
    cache.clear()
    assert cache.nbytes == 0 and len(cache) == 0


def _make_idle(root, experiment):
    """Set every file mtime of the experiment in the past (but keep directory mtimes)."""
    old = time.time() - 3600
    for dirpath, _, filenames in os.walk(root / experiment):
        for name in filenames:
            os.utime(os.path.join(dirpath, name), (old, old))


def test_listing_quick_key(make_logs):
    _make_idle(make_logs, "healthy")
    library = Library(make_logs)
    library.revalidate_s = 3600
    summary = next(s for s in library.list_summaries() if s["id"] == "healthy")
    assert summary["progress"] == 1.0
    for name in ("test.csv", "train.csv", "training_data.csv"):  # In-place rewrites: no directory change
        table = make_logs / "healthy" / "run-0" / name
        lines = table.read_text().splitlines()
        table.write_text("\n".join(lines[:3]) + "\n")
    # Listings trust the cached summary of an idle experiment for up to `revalidate_s`...
    assert next(s for s in library.list_summaries() if s["id"] == "healthy")["status"] == "COMPLETED"
    # ...while a direct access always checks every file.
    assert library.get("healthy").runs[0].status == "CANCELLED"  # type: ignore[union-attr]
    assert next(s for s in library.list_summaries() if s["id"] == "healthy")["status"] == "CANCELLED"


def test_listing_quick_key_sees_structural_changes_and_expires(make_logs):
    _make_idle(make_logs, "healthy")
    library = Library(make_logs)
    library.revalidate_s = 3600
    library.list_summaries()
    raw = json.loads((make_logs / "healthy" / "experiment.json").read_text())
    raw["n_steps"] = 20_000
    (make_logs / "healthy" / "experiment.json").write_text(json.dumps(raw))
    assert next(s for s in library.list_summaries() if s["id"] == "healthy")["n_steps"] == 20_000
    (make_logs / "healthy" / "run-1" / "pid").write_text("999999999")  # New file: the run directory changes
    (make_logs / "healthy" / "run-2" / "test.csv").unlink()
    summary = next(s for s in library.list_summaries() if s["id"] == "healthy")
    assert summary["health"] == "warning"  # missing-table on run-2
    # Expiry: appends are seen once the last full check is older than `revalidate_s`
    _make_idle(make_logs, "healthy")
    library.list_summaries()
    with open(make_logs / "healthy" / "run-0" / "test.csv", "a") as f:
        f.write("0,0.5,10,1.0,1700000500.0,15000\n")
    library.revalidate_s = 0
    run0 = library.get("healthy").runs[0]  # type: ignore[union-attr]
    assert run0.latest_step == 15_000


def test_hot_experiments_are_always_fully_checked(make_logs):
    library = Library(make_logs)  # Fixture files were just written: every experiment is hot
    library.revalidate_s = 3600
    library.list_summaries()
    with open(make_logs / "healthy" / "run-0" / "test.csv", "a") as f:
        f.write("0,0.5,10,1.0,1700000500.0,15000\n")
    summary = next(s for s in library.list_summaries() if s["id"] == "healthy")
    assert summary["progress"] == 1.0
    assert library.get("healthy").runs[0].latest_step == 15_000  # type: ignore[union-attr]


@pytest.mark.parametrize("resolution", [1, 3, 10, 1000])
def test_fast_bucketing_matches_polars(resolution):
    rng = np.random.default_rng(resolution)
    x = np.sort(rng.integers(0, 10_000, 3000)).astype(float)
    x[:10] = [0.5, 1.5, 2.5, 2.5, 3.5, 4.5, 5, 5, 5, 6]  # half-to-even rounding cases
    x.sort()
    y = rng.random(3000)
    xb, mean = _bucket_sorted(x, y, resolution)
    expected = bucket(x, y, resolution)
    assert xb is not None
    np.testing.assert_array_equal(xb, expected["xb"].to_numpy())
    np.testing.assert_allclose(mean, expected["y"].to_numpy(), rtol=1e-12)


def test_bucket_runs_handles_unsorted_runs():
    x_sorted, x_unsorted = np.array([0.0, 1, 2, 3]), np.array([3.0, 1, 2, 0])
    y = np.array([1.0, 2, 3, 4])
    assert _bucket_sorted(x_unsorted, y, 1) == (None, None)
    stacked = bucket_runs([(x_sorted, y), (x_unsorted, y)], 2).sort("run", "xb")
    # x / 2 = [0, 0.5, 1, 1.5] rounds (half to even) to [0, 0, 1, 2]: buckets 0 <- {0, 1}, 2 <- {2}, 4 <- {3}
    assert stacked.filter(stacked["run"] == 0)["y"].to_list() == [1.5, 3.0, 4.0]
    run1 = stacked.filter(stacked["run"] == 1).sort("xb")
    assert run1["xb"].to_list() == [0.0, 2.0, 4.0] and run1["y"].to_list() == [3.0, 3.0, 1.0]


def test_distinct_and_auto_resolution_on_sorted_and_unsorted_x():
    sorted_x = np.repeat(np.arange(0, 10_001, 1000.0), 3)
    assert list(_distinct(sorted_x)) == list(np.arange(0, 10_001, 1000.0))  # type: ignore[arg-type]
    assert _distinct(np.arange(5000.0)) is None
    assert list(_distinct(np.array([3.0, 1, 3, 2]))) == [1, 2, 3]  # type: ignore[arg-type]
    assert auto_resolution([sorted_x, sorted_x[::-1]]) == 1000
    assert auto_resolution([np.arange(0, 100_000.0)]) == 200
