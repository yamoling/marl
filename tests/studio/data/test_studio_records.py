"""Discovery, records, issues, capabilities, status and PID ownership."""

import json
import os
import shutil

import pytest
from conftest import SCENARIOS, run_frames

from studio.backend.data.library import Library
from studio.backend.data.records import PidState, build_experiment, check_pid, discover
from studio.backend.data.sources.csv import tail_value

EXPECTED = {
    # scenario: (issue codes, params capability, experiment status)
    "healthy": (set(), "full", "COMPLETED"),
    "light": ({"missing-keys"}, "partial", "COMPLETED"),
    "unknown-class": (set(), "full", "COMPLETED"),
    "corrupt-run": ({"invalid-run-json", "seed-inferred"}, "full", "COMPLETED"),
    "missing-table": ({"missing-table"}, "full", "COMPLETED"),
    "extra-table": (set(), "full", "COMPLETED"),
    "late-column": (set(), "full", "COMPLETED"),
    "no-experiment-json": ({"missing-experiment-json"}, "none", "COMPLETED"),
    "running": (set(), "full", "RUNNING"),
    "jsonl": (set(), "full", "COMPLETED"),
    "sqlite": (set(), "full", "COMPLETED"),
}


@pytest.fixture(scope="module")
def library(fixture_logs):
    return Library(fixture_logs)


def test_discovery_finds_every_fixture(fixture_logs):
    assert sorted(p.name for p in discover(fixture_logs)) == sorted(SCENARIOS)


def test_discovery_depth_and_skips(tmp_path):
    (tmp_path / "a" / "b" / "exp" / "run-0").mkdir(parents=True)
    (tmp_path / "a" / "b" / "exp" / "run-0" / "run.json").write_text("{}")
    (tmp_path / "a" / "b" / "c" / "too-deep" / "run-0").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "too-deep" / "run-0" / "test.csv").write_text("time_step\n0\n")
    (tmp_path / ".hidden" / "run-0").mkdir(parents=True)
    (tmp_path / ".hidden" / "run-0" / "run.json").write_text("{}")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "x.txt").write_text("")
    assert [p.relative_to(tmp_path).as_posix() for p in discover(tmp_path)] == ["a/b/exp"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_issue_codes_and_capabilities(library, scenario):
    codes, params, status = EXPECTED[scenario]
    record = library.get(scenario)
    assert record is not None
    assert {i.code for i in record.all_issues} == codes
    assert record.capabilities.params == params
    assert record.capabilities.metrics is True
    assert record.capabilities.launch is None and record.capabilities.replay is None
    assert record.status == status


def test_corrupt_run_is_kept_with_an_error(library):
    record = library.get("corrupt-run")
    run = record.run("run-1")
    assert run is not None and run.seed == 1 and run.raw == {}
    assert [i.code for i in run.issues if i.level == "error"] == ["invalid-run-json"]
    assert run.issues[0].scope == "run:corrupt-run/run-1"
    assert library.detail("corrupt-run")["health"] == "error"


def test_missing_table_flags_the_run(library):
    record = library.get("missing-table")
    issue = next(i for i in record.all_issues if i.code == "missing-table")
    assert issue.scope == "run:missing-table/run-2" and issue.path == "test"


def test_run_fields_and_latest_step(library):
    record = library.get("healthy")
    assert [r.id for r in record.runs] == ["healthy/run-0", "healthy/run-1", "healthy/run-2"]
    run = record.runs[0]
    expected = max(int(df["time_step"].max()) for df in run_frames(0).values())  # type: ignore[arg-type]
    assert run.latest_step == expected
    assert run.status == "COMPLETED" and run.progress == 1.0 and run.pid is None
    assert run.to_json()["config"] == {"n_tests": 2, "test_interval": 1000, "save_weights": False, "save_actions": True}


def test_summary_and_detail_shapes(library):
    summaries = library.list_summaries()
    assert len(summaries) == len(SCENARIOS)
    keys = {"id", "name", "algo", "env", "created", "n_steps", "status", "progress", "health", "issue_counts", "n_runs", "running_runs"}
    assert all(set(s) == keys for s in summaries)
    healthy = next(s for s in summaries if s["id"] == "healthy")
    assert healthy["algo"] == "QMix" and healthy["env"] == "LLE-lvl6" and healthy["n_steps"] == 10_000
    assert healthy["issue_counts"] == {"info": 0, "warning": 0, "error": 0}
    detail = library.detail("healthy")
    assert set(detail) == keys | {"raw", "issues", "capabilities", "runs", "params"}
    json.dumps(detail)  # JSON-serializable
    assert library.detail("../outside") is None and library.detail("/etc") is None and library.detail("nope") is None


def test_summary_filters(library):
    assert {s["id"] for s in library.list_summaries(status="RUNNING")} == {"running"}
    assert {s["id"] for s in library.list_summaries(health="error")} == {"corrupt-run"}
    assert {s["id"] for s in library.list_summaries(algo="none")} == set()
    assert len(library.list_summaries(algo="qmix")) == len(SCENARIOS) - 2  # light and no-experiment-json


def test_running_process_ownership_is_verified(fixture_logs):
    record = build_experiment(fixture_logs / "running", fixture_logs)
    run = record.runs[0]
    assert run.status == "RUNNING" and run.pid is not None
    assert record.running_runs == 1
    assert run.to_json()["pid"] == run.pid


def test_pid_of_foreign_process_is_unverifiable(make_logs):
    rundir = make_logs / "healthy" / "run-0"
    (rundir / "pid").write_text(str(os.getpid()))
    state = check_pid(rundir, make_logs / "healthy", make_logs)
    assert state.state == "unverifiable"
    record = Library(make_logs).get("healthy")
    assert "pid-unverifiable" in {i.code for i in record.all_issues}
    assert record.runs[0].status == "COMPLETED"


def test_stale_pid_file_is_ignored_and_kept(make_logs):
    rundir = make_logs / "healthy" / "run-0"
    (rundir / "pid").write_text("999999999")
    assert check_pid(rundir, make_logs / "healthy", make_logs) == PidState(None, "none")
    assert (rundir / "pid").exists()


def test_pid_checker_is_injectable(fixture_logs):
    library = Library(fixture_logs, pid_checker=lambda rundir, expdir, root: PidState(42, "verified"))
    # Only runs having a pid file are checked.
    assert library.get("healthy").status == "COMPLETED"
    record = library.get("running")
    assert record.runs[0].pid == 42 and record.status == "RUNNING"


def test_logdir_mismatch_after_move(make_logs):
    shutil.move(make_logs / "healthy", make_logs / "moved")
    record = Library(make_logs).get("moved")
    mismatches = [i for i in record.all_issues if i.code == "logdir-mismatch"]
    assert len(mismatches) == 1 + len(record.runs)
    assert all(i.level == "info" for i in mismatches)
    assert record.runs[0].tables["test"].file.parent == make_logs / "moved" / "run-0"


def test_seed_inferred_from_dirname(make_logs):
    run_json = make_logs / "healthy" / "run-1" / "run.json"
    raw = json.loads(run_json.read_text())
    del raw["seed"]
    run_json.write_text(json.dumps(raw))
    record = Library(make_logs).get("healthy")
    assert record.run("run-1").seed == 1
    assert [i.code for i in record.run("run-1").issues] == ["seed-inferred"]


def test_status_created_and_cancelled(make_logs):
    exp = make_logs / "healthy"
    (exp / "run-9").mkdir()
    (exp / "run-9" / "run.json").write_text(json.dumps({"seed": 9, "n_steps": 10_000}))
    lines = (exp / "run-1" / "test.csv").read_text().splitlines()
    (exp / "run-1" / "test.csv").write_text("\n".join(lines[:5]) + "\n")
    for name in ("train.csv", "training_data.csv"):
        (exp / "run-1" / name).unlink()
    record = Library(make_logs).get("healthy")
    assert record.run("run-9").status == "CREATED" and record.run("run-9").latest_step is None
    assert record.run("run-1").status == "CANCELLED" and record.run("run-1").latest_step == 1000
    assert record.status == "CANCELLED"


def test_cache_invalidation_on_file_change(make_logs):
    library = Library(make_logs)
    first = library.get("healthy")
    assert library.get("healthy") is first
    with open(make_logs / "healthy" / "run-0" / "test.csv", "a") as f:
        f.write("0,0.5,10,1.0,1700000500.0,12000\n")
    second = library.get("healthy")
    assert second is not first and second.runs[0].latest_step == 12_000


def test_tail_value_ignores_partial_last_line(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,time_step\n" + "".join(f"{i},{i * 10}\n" for i in range(2000)) + "5,99")
    assert tail_value(path, "time_step") == 19_990
    path.write_text("a,time_step\n")
    assert tail_value(path, "time_step") is None
    assert tail_value(path, "missing") is None


def test_empty_and_unreadable_tables(make_logs):
    rundir = make_logs / "healthy" / "run-0"
    (rundir / "empty.csv").write_text("")
    (rundir / "header-only.csv").write_text("x,time_step\n")
    (rundir / "locked.csv").write_text("x,time_step\n1,2\n")
    (rundir / "locked.csv").chmod(0)
    try:
        record = Library(make_logs).get("healthy")
    finally:
        (rundir / "locked.csv").chmod(0o644)
    tables = record.runs[0].tables
    assert tables["empty"].columns == {} and tables["header-only"].columns == {"x": "num", "time_step": "num"}
    if os.geteuid() != 0:
        assert "locked" not in tables
        issue = next(i for i in record.runs[0].issues if i.code == "unreadable-table")
        assert issue.scope == "table:healthy/run-0/locked" and issue.detail is not None
    assert record.capabilities.metrics and record.runs[0].status == "COMPLETED"
