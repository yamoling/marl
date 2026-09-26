"""Series computation: golden test against the legacy pipeline and the robustness rules."""

import numpy as np
import pytest
from conftest import N_STEPS

from studio.backend.data.library import Library
from studio.backend.data.series import SeriesQuery, auto_resolution, compute, m4_indices, nice_number

CATEGORY_TABLES = {"Test": "test", "Train": "train", "Training data": "training_data"}


@pytest.fixture(scope="module")
def library(fixture_logs):
    return Library(fixture_logs)


def q(experiment="healthy", table="test", metric="score-0", **kwargs) -> SeriesQuery:
    return SeriesQuery(experiment, table, metric, **kwargs)


@pytest.mark.parametrize("granularity", [1000, 250])
def test_golden_against_legacy_results(library, fixture_logs, granularity):
    from marl.models.experiment import LightExperiment

    legacy = LightExperiment.load(fixture_logs / "healthy").get_results_datasets(granularity)
    record = library.get("healthy")
    checked = 0
    for dataset in legacy:
        if dataset.label == "episode_num":
            continue
        table = CATEGORY_TABLES[dataset.category]
        results = {
            band: compute(record, q(table=table, metric=dataset.label, resolution=granularity, band=band, max_points=5000))
            for band in ("std", "minmax", "ci95")
        }
        std, minmax, ci95 = results["std"], results["minmax"], results["ci95"]
        assert std.resolution == granularity
        np.testing.assert_array_equal(std.x, dataset.ticks)
        centre = np.array(std.center, dtype=float)
        tol = {"rtol": 1e-6, "atol": 1e-6}
        np.testing.assert_allclose(centre, dataset.mean, **tol)
        np.testing.assert_allclose(np.array(std.hi, dtype=float) - centre, dataset.std, **tol)
        np.testing.assert_allclose(np.array(minmax.lo, dtype=float), dataset.min, **tol)
        np.testing.assert_allclose(np.array(minmax.hi, dtype=float), dataset.max, **tol)
        np.testing.assert_allclose(np.array(ci95.hi, dtype=float) - centre, dataset.ci95, **tol)
        assert set(std.n) == {3}
        checked += 1
    assert checked == 4 + 2 + 3  # test, train (without episode_num) and training_data metrics


def test_late_column_keeps_other_metrics(library):
    record = library.get("late-column")
    other = compute(record, q("late-column", metric="score-0"))
    late = compute(record, q("late-column", metric="late-metric"))
    assert len(other.x) == N_STEPS // 20 + 1 and other.x[0] == 0
    assert late.x[0] == 6000 and late.x[-1] == N_STEPS
    assert late.used_runs == ["late-column/run-0", "late-column/run-1"] and late.missing_runs == []


def test_missing_table_populates_missing_runs(library):
    result = compute(library.get("missing-table"), q("missing-table"))
    assert result.missing_runs == ["missing-table/run-2"]
    assert result.used_runs == ["missing-table/run-0", "missing-table/run-1"]
    assert set(result.n) == {2}


def test_run_subset_and_unknown_runs(library):
    result = compute(library.get("healthy"), q(runs=("healthy/run-1", "run-2", "healthy/run-7"), include_runs=True))
    assert result.used_runs == ["healthy/run-1", "healthy/run-2"]
    assert result.missing_runs == ["healthy/run-7"]
    assert [i.code for i in result.issues] == ["unknown-run"]
    assert [r.seed for r in result.runs] == [1, 2]


def test_wall_time_is_relative_to_each_run(library):
    result = compute(library.get("healthy"), q(x="wall_time", include_runs=True, resolution=1))
    assert all(r.x[0] == 0 for r in result.runs)
    # Runs start one hour apart: a global origin would give x up to ~7300 s.
    assert max(result.x) < 200
    assert set(result.n) == {3}


def test_auto_resolution_exact_for_test_tables(library):
    result = compute(library.get("healthy"), q())
    assert result.resolution == 1000
    assert result.x == [float(s) for s in range(0, N_STEPS + 1, 1000)]


def test_auto_resolution_about_500_buckets_for_train_tables(library):
    result = compute(library.get("healthy"), q(table="train", max_points=5000))
    assert result.resolution == nice_number(N_STEPS / 500)
    assert 400 <= len(result.x) <= 600


def test_auto_resolution_helpers():
    assert nice_number(0.3) == 1
    assert nice_number(20) == 20
    assert nice_number(2900) == 2000
    assert nice_number(4000) == 5000
    assert auto_resolution([np.array([0.0, 5000, 10000]), np.array([0.0, 5000])]) == 5000
    assert auto_resolution([]) == 1


def test_downsampling_keeps_global_extrema(library):
    record = library.get("healthy")
    full = compute(record, q(table="training_data", metric="td-loss", resolution=5, max_points=5000))
    small = compute(record, q(table="training_data", metric="td-loss", resolution=5, max_points=100))
    assert len(full.x) > 1000 and len(small.x) <= 100
    assert min(small.center) == min(full.center) and max(small.center) == max(full.center)  # type: ignore[type-var]
    assert len(small.lo) == len(small.hi) == len(small.n) == len(small.x)  # type: ignore[arg-type]


def test_m4_indices():
    y = np.sin(np.linspace(0, 20, 10_000))
    y[1234], y[8765] = 5, -5
    idx = m4_indices(y, 200)
    assert len(idx) <= 200 and 1234 in idx and 8765 in idx and idx[0] == 0 and idx[-1] == 9999
    assert list(m4_indices(y[:10], 200)) == list(range(10))


def test_bool_columns_are_plottable(library):
    record = library.get("extra-table")
    catalog = library.catalog("extra-table")
    assert "agent-0-exited" in catalog["tables"]["test-policy-on-test-envs"]["metrics"]
    result = compute(record, q("extra-table", table="test-policy-on-test-envs", metric="agent-0-exited", include_runs=True))
    assert result.used_runs == ["extra-table/run-0", "extra-table/run-1"]
    assert all(set(r.y) <= {0.0, 1.0} or 0 <= min(r.y) <= max(r.y) <= 1 for r in result.runs)  # type: ignore[type-var]
    assert 0 <= result.center[0] <= 1  # type: ignore[index]


@pytest.mark.parametrize("scenario", ["jsonl", "sqlite"])
def test_other_sources_match_csv(library, scenario):
    expected = compute(library.get("healthy"), q())
    result = compute(library.get(scenario), q(scenario))
    assert result.x == expected.x and result.n == expected.n
    np.testing.assert_allclose(result.center, expected.center)  # type: ignore[arg-type]
    np.testing.assert_allclose(result.hi, expected.hi)  # type: ignore[arg-type]


def test_center_none_and_band_none(library):
    record = library.get("healthy")
    runs_only = compute(record, q(center="none"))
    assert runs_only.center is None and runs_only.lo is None and len(runs_only.runs) == 3
    no_band = compute(record, q(band="none", center="median"))
    assert no_band.lo is None and no_band.center is not None


def test_batch_partial_success(library):
    out = library.series(
        [
            {"experiment": "healthy", "table": "test", "metric": "score-0"},
            {"experiment": "does-not-exist", "table": "test", "metric": "score-0"},
            {"experiment": "healthy", "table": 3, "metric": "score-0"},
            {"experiment": "healthy", "table": "test", "metric": "score-0", "band": "wide"},
            {"experiment": "healthy", "table": "test", "metric": "nope"},
        ]
    )
    assert [o["ok"] for o in out] == [True, False, False, False, True]
    assert out[1]["issue"]["code"] == "unknown-experiment"
    assert out[2]["issue"]["code"] == out[3]["issue"]["code"] == "invalid-query"
    assert out[4]["result"]["missing_runs"] == ["healthy/run-0", "healthy/run-1", "healthy/run-2"]
    assert set(out[0]["result"]) == {"x", "center", "lo", "hi", "n", "runs", "used_runs", "missing_runs", "resolution", "issues"}


def test_query_from_json_defaults():
    query = SeriesQuery.from_json({"experiment": "e", "table": "t", "metric": "m", "center": "none", "max_points": 99_999})
    assert query.include_runs is True and query.max_points == 5000 and query.runs is None
    with pytest.raises(ValueError):
        SeriesQuery.from_json({"experiment": "e", "table": "t", "metric": "m", "smoothing": 0.5})


def test_results_follow_file_changes(make_logs):
    library = Library(make_logs)
    before = compute(library.get("healthy"), q())
    with open(make_logs / "healthy" / "run-0" / "test.csv", "a") as f:
        f.write("0,0.5,10,100.0,1700000500.0,11000\n")
    after = compute(library.get("healthy"), q())
    assert after.x[-1] == 11_000 and len(after.x) == len(before.x) + 1 and after.n[-1] == 1


def test_preview(library):
    preview = library.preview("healthy", points=8)
    assert preview["metric"] == {"table": "test", "metric": "score-0"}
    assert 0 < len(preview["result"]["x"]) <= 8
    assert library.preview("nope") is None


def test_test_steps_and_episodes(library):
    assert library.test_steps("healthy") == list(range(0, N_STEPS + 1, 1000))
    assert library.test_steps("sqlite") == list(range(0, N_STEPS + 1, 1000))
    episodes = library.episodes("healthy", 1000)
    assert [(e["run"], e["test"]) for e in episodes] == [(f"healthy/run-{s}", t) for s in range(3) for t in range(2)]
    assert all(e["has_actions"] for e in episodes)
    assert set(episodes[0]["metrics"]) == {"gems_collected", "exit_rate", "episode_len", "score-0"}
    assert not any(e["has_actions"] for e in library.episodes("healthy", 2000))
