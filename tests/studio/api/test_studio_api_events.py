"""Server-Sent Events: snapshot, run-progress, experiment-added/removed/changed and ping."""

import json
import shutil
import threading
import time

import orjson
from conftest import build_logs, healthy_raw, make_client, stop_processes, write_experiment, write_run


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], orjson.loads(lines["data"])))
    return events


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


class Stream:
    """Consume `/api/events` in a background thread until the hub closes the stream."""

    def __init__(self, http):
        self.http = http
        self.hub = http.app.state.events
        self.response = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.response = self.http.get("/api/events")

    def __enter__(self):
        self.thread.start()
        wait_for(lambda: self.hub.n_subscribers == 1 and self.hub.initialized)
        return self

    def close(self) -> list[tuple[str, dict]]:
        self.hub.close()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
        assert self.response is not None and self.response.status_code == 200
        assert self.response.headers["content-type"].startswith("text/event-stream")
        return parse_sse(self.response.text)

    def __exit__(self, *exc):
        if self.thread.is_alive():
            self.hub.close()
            self.thread.join(timeout=10)


def test_event_stream(make_logs):
    http = make_client(make_logs)
    hub = http.app.state.events
    train = make_logs / "running" / "run-0" / "train.csv"
    with Stream(http) as stream:
        assert hub._known is not None
        initial = hub._known["healthy"]
        with open(train, "a") as f:
            f.write("9999,0.5,1.0,1700000100.0,9990\n")
        wait_for(lambda: any(p["latest_step"] == 9990 for p in hub._running.values()))
        exp = make_logs / "new-exp"
        write_experiment(exp, healthy_raw(exp))
        write_run(exp, 0)
        raw = json.loads((make_logs / "healthy" / "experiment.json").read_text())
        raw["n_steps"] = 20_000
        time.sleep(0.01)
        (make_logs / "healthy" / "experiment.json").write_text(json.dumps(raw))
        wait_for(lambda: hub._known is not None and "new-exp" in hub._known and hub._known["healthy"] != initial)
        time.sleep(0.5)  # Leave time for the scan events and a ping
        events = stream.close()
    names = [name for name, _ in events]
    assert names[0] == "snapshot"
    snapshot = events[0][1]["running"]
    assert [(r["experiment"], r["run"], r["status"]) for r in snapshot] == [("running", "running/run-0", "RUNNING")]
    progress = [data for name, data in events if name == "run-progress"]
    assert progress and progress[-1] == {
        "experiment": "running",
        "run": "running/run-0",
        "status": "RUNNING",
        "progress": 0.999,
        "latest_step": 9990,
    }
    assert ("experiment-added", {"experiment": "new-exp"}) in events
    assert ("experiment-changed", {"experiment": "healthy"}) in events
    assert "ping" in names
    assert hub.n_subscribers == 0 and hub._task is None


def test_final_progress_when_a_run_stops_and_removal(tmp_path):
    processes = build_logs(tmp_path / "logs")
    try:
        http = make_client(tmp_path / "logs")
        hub = http.app.state.events
        with Stream(http) as stream:
            stop_processes(processes)
            wait_for(lambda: not hub._running)
            shutil.rmtree(tmp_path / "logs" / "light")
            wait_for(lambda: hub._known is not None and "light" not in hub._known)
            events = stream.close()
    finally:
        stop_processes(processes)
    progress = [data for name, data in events if name == "run-progress"]
    assert progress[-1]["run"] == "running/run-0" and progress[-1]["status"] != "RUNNING"
    assert ("experiment-removed", {"experiment": "light"}) in events


def test_poller_stops_with_last_client(api):
    hub = api.app.state.events
    with Stream(api) as stream:
        assert hub._task is not None
        events = stream.close()
    assert events[0][0] == "snapshot"
    assert hub._task is None and hub.n_subscribers == 0
