"""Accepted launches are monitored until exit, with persistent run-scoped failures."""

import json
import subprocess
import threading
import time

import pytest
from conftest import make_client, run_frames, run_raw, write_run

from studio.backend.services import launcher


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("launcher monitor did not finish")
        time.sleep(0.01)


class LateProcess:
    def __init__(self, code, stderr, output):
        self.code = code
        self.stderr = stderr
        self.output = output
        self.exit = threading.Event()

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("start_run", timeout)
        assert self.exit.wait(5)
        self.output.write(self.stderr)
        self.output.flush()
        return self.code


def test_partial_late_failure_survives_new_client_and_restart_clears(mut_api, monkeypatch):
    http, root = mut_api
    exp = root / "healthy"
    processes = []
    events = []
    monkeypatch.setattr(http.app.state.events, "publish_threadsafe", lambda name, data: events.append((name, data)))

    def popen(*args, **kwargs):
        process = LateProcess(3, b"late traceback", kwargs["stderr"])
        processes.append(process)
        return process

    monkeypatch.setattr(launcher, "_popen", popen)
    response = http.post("/api/experiments/healthy/runs", json={"n_runs": 2, "seed": 5, "n_tests": 1})
    assert response.status_code == 202
    write_run(exp, 5)
    processes[0].exit.set()
    wait_for(lambda: bool(events))
    name, payload = events[0]
    assert name == "launch-failed" and payload["runs"] == ["healthy/run-6"]
    assert payload["issue"]["scope"] == "run:healthy/run-6"
    assert "late traceback" in payload["issue"]["detail"]
    detail = make_client(root).get("/api/experiments/healthy").json()
    assert detail["health"] == "error"
    assert not any(i["code"] == "launch-failed" for i in next(r for r in detail["runs"] if r["seed"] == 5)["issues"])
    assert any(i["code"] == "launch-failed" for i in next(r for r in detail["runs"] if r["seed"] == 6)["issues"])

    # Run 6 can be restarted even without a run.json; its seed is inferred from the dirname.
    processes.clear()
    response = http.post("/api/runs/healthy/run-6/restart")
    assert response.status_code == 202, response.text
    rundir = exp / "run-6"
    (rundir / "run.json").write_text(json.dumps(run_raw(rundir, 6)))
    run_frames(6)["test"].write_csv(rundir / "test.csv")
    processes[0].code = 0
    processes[0].stderr = b""
    processes[0].exit.set()
    wait_for(lambda: not (exp / "run-6" / launcher.LAUNCH_FAILURE_FILE).exists())
    detail = make_client(root).get("/api/experiments/healthy").json()
    assert not any(i["code"] == "launch-failed" for i in detail["issues"])


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("stderr", [b"empty run warning", b""])
def test_zero_exit_without_data_is_persisted(mut_api, monkeypatch, early, stderr):
    http, root = mut_api
    events = []
    monkeypatch.setattr(http.app.state.events, "publish_threadsafe", lambda name, data: events.append((name, data)))
    processes = []

    def popen(*args, **kwargs):
        process = LateProcess(0, stderr, kwargs["stderr"])
        processes.append(process)
        if early:
            process.output.write(process.stderr)
            process.output.flush()
            process.wait = lambda timeout=None: 0
        return process

    monkeypatch.setattr(launcher, "_popen", popen)
    assert http.post("/api/experiments/healthy/runs", json={"n_runs": 1, "seed": 9, "n_tests": 1}).status_code == 202
    processes[0].exit.set()
    wait_for(lambda: bool(events))
    assert events[0][1]["runs"] == ["healthy/run-9"]
    assert (stderr.decode() or "exit 0, no usable run data") in events[0][1]["issue"]["detail"]
    detail = make_client(root).get("/api/experiments/healthy").json()
    run = next(r for r in detail["runs"] if r["seed"] == 9)
    assert run["issues"][0]["code"] in ("invalid-run-json", "seed-inferred")
    assert any(i["code"] == "launch-failed" and i["scope"] == "run:healthy/run-9" for i in run["issues"])
