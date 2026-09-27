"""Lazy launch/replay capability checks (full deserialization with marl)."""

import time

import pytest

from studio.backend.data import health
from studio.backend.data.library import Library


@pytest.fixture(scope="module")
def library(fixture_logs):
    return Library(fixture_logs)


def test_healthy_is_launchable(library):
    ok, issues = health.check_launchable(library.get("healthy"))
    assert ok is True and issues == []


@pytest.mark.parametrize(("scenario", "path"), [("light", "trainer"), ("unknown-class", "trainer.mixer")])
def test_degraded_experiments_are_not_launchable(library, scenario, path):
    record = library.get(scenario)
    ok, issues = health.check_launchable(record)
    assert ok is False
    assert [i.code for i in issues] == ["deserialize-failed"]
    assert issues[0].path == path and issues[0].detail is not None
    if scenario == "light":
        assert "missing-keys" in {i.code for i in record.issues}


def test_no_experiment_json_is_not_launchable(library):
    assert health.check_launchable(library.get("no-experiment-json")) == (False, [])


def test_library_health_updates_capabilities(fixture_logs):
    library = Library(fixture_logs)
    assert library.detail("unknown-class")["capabilities"]["launch"] is None
    out = library.health("unknown-class")
    assert out["capabilities"] == {"metrics": True, "params": "full", "replay": False, "launch": False}
    assert "deserialize-failed" in {i["code"] for i in out["issues"]}
    detail = library.detail("unknown-class")
    assert detail["capabilities"]["launch"] is False and detail["health"] == "error"
    assert library.health("healthy")["capabilities"]["launch"] is True


def test_error_path_extraction():
    raw = {"trainer": {"memory": {"class-name": "Mem"}, "class-name": "DQN"}, "class-name": "Experiment"}
    assert health.error_path(KeyError("Missing value for required field lr of class DQN"), raw) == "trainer.lr"
    assert health.error_path(KeyError("Missing value for required field env of class Experiment"), raw) == "env"
    assert health.error_path(KeyError("Unknown subclass Mem for Memory"), raw) == "trainer.memory"
    assert health.error_path(ValueError("boom"), raw) is None
    light = {"n_steps": 1, "class-name": "LightExperiment"}
    assert health.error_path(KeyError("Missing value for required field trainer of class Experiment"), light) == "trainer"
    assert health.error_path(KeyError("Missing value for required field n_steps of class Experiment"), light) is None


def test_results_are_cached_and_timeouts_reported(library):
    calls = []

    def loader(path):
        calls.append(path)

    record = library.get("healthy")
    assert health.check_launchable(record, loader=loader) == (True, [])
    assert health.check_launchable(record, loader=loader) == (True, [])
    assert len(calls) == 1

    def slow(path):
        time.sleep(1)

    ok, issues = health.check_launchable(record, timeout=0.05, loader=slow)
    assert ok is False and issues[0].code == "deserialize-failed" and "TimeoutError" in (issues[0].detail or "")
