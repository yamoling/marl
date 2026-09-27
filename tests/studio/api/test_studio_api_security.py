"""
Security regression tests, ported from tests/test_ui_security.py and tests/test_ui_run_paths.py:
Host/Origin checks, path confinement, PID ownership, SPA confinement, loopback binding.
"""

import os
import signal
import time

import psutil
import pytest
from conftest import make_client

from studio.backend import security
from studio.backend.data.library import Library
from studio.backend.services import launcher, runs

ESCAPE_ROUTES = [
    ("get", "/api/experiments/escape"),
    ("get", "/api/experiments/escape/catalog"),
    ("get", "/api/experiments/escape/preview"),
    ("get", "/api/experiments/escape/test-steps"),
    ("get", "/api/experiments/escape/episodes?step=0"),
    ("get", "/api/experiments/escape/launch-defaults"),
    ("post", "/api/experiments/escape/health"),
    ("post", "/api/experiments/escape/runs"),
    ("post", "/api/experiments/escape/stop"),
    ("patch", "/api/experiments/escape"),
    ("delete", "/api/experiments/escape"),
    ("get", "/api/runs/escape/run-0/replay?step=0&test=0"),
    ("post", "/api/runs/escape/run-0/stop"),
    ("post", "/api/runs/escape/run-0/restart"),
]


@pytest.fixture
def escape(mut_api, tmp_path, monkeypatch):
    """A symlink below the logs root pointing to an experiment outside of it."""
    http, root = mut_api
    outside = tmp_path / "outside"
    (outside / "run-0").mkdir(parents=True)
    (outside / "experiment.json").write_text("{}")
    (outside / "run-0" / "run.json").write_text("{}")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    return http, root, outside


@pytest.mark.parametrize(("method", "url"), ESCAPE_ROUTES)
def test_paths_outside_root_rejected_before_loading(escape, monkeypatch, method, url):
    http, _, outside = escape
    monkeypatch.setattr(Library, "get", lambda *a, **kw: pytest.fail("must reject before loading"))
    kwargs = {"json": {"n_runs": 1, "n_tests": 1, "seed": 1, "new_id": "x"}} if method in ("post", "patch") else {}
    response = getattr(http, method)(url, **kwargs)
    assert response.status_code == 403
    assert set(response.json()) == {"error", "message"}
    assert (outside / "experiment.json").exists()


def test_malformed_root_and_traversal_ids(mut_api):
    http, _ = mut_api
    assert http.get("/api/experiments/missing").status_code == 404
    assert http.get("/api/experiments/%2E").status_code == 400
    assert http.get("/api/experiments//etc").status_code == 400
    assert http.get("/api/experiments/healthy%2F..%2F..%2Foutside").status_code == 403
    assert http.get("/api/experiments/a%5Cb").status_code == 400


def test_rename_rejects_escaping_destinations(escape):
    http, root, outside = escape
    for new_id in ("../renamed", "/tmp/renamed", "escape/new", "healthy/nested"):
        response = http.patch("/api/experiments/healthy", json={"new_id": new_id})
        assert response.status_code in (400, 403), new_id
    assert http.patch("/api/experiments/healthy", json={"new_id": "escape/new"}).status_code == 403
    assert http.patch("/api/experiments/healthy", json={"new_id": "missing/nested"}).status_code == 404
    assert http.patch("/api/experiments/healthy", json={}).status_code == 400
    assert (root / "healthy").is_dir() and not (outside / "new").exists()


def test_cross_origin_mutations_are_blocked_but_local_dev_origins_work(api):
    body = {"queries": [{"experiment": "healthy", "table": "test", "metric": "score-0"}]}
    for headers in (
        {"Origin": "https://evil.example"},
        {"Origin": "http://localhost.evil.example"},
        {"Origin": "http://localhost:bad"},
        {"Origin": "http://localhost:9000"},
        {"Origin": "http://localhost:5180"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "cross-site"},
    ):
        response = api.post("/api/series", json=body, headers=headers)
        assert response.status_code == 403, headers
        assert response.json()["error"] == "forbidden"
    for origin in ("http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://localhost:5000"):
        assert api.post("/api/series", json=body, headers={"Origin": origin}).status_code == 200, origin
    preflight = api.options("/api/series", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert preflight.headers.get("access-control-allow-origin") is None
    ok = api.options("/api/series", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_host_must_be_local(api):
    assert api.get("/", headers={"Host": "attacker.example"}).status_code == 403
    assert api.get("/api/experiments", headers={"Host": "attacker.example"}).status_code == 403
    assert api.post("/api/series", json={}, headers={"Host": "localhost.evil"}).status_code == 403
    assert api.options("/api/series", headers={"Host": "attacker.example", "Origin": "http://localhost:5173"}).status_code == 403
    assert api.get("/api/health", headers={"Host": "127.0.0.1:5000"}).status_code == 200
    assert api.get("/api/health", headers={"Host": "[::1]:5000"}).status_code == 200


def test_spa_is_confined_to_dist(fixture_logs, tmp_path):
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>safe</html>")
    (dist / "assets" / "app.js").write_text("js")
    (dist / "leak").symlink_to(secret)
    http = make_client(fixture_logs, dist_dir=dist)
    assert http.get("/").text == "<html>safe</html>"
    assert http.get("/assets/app.js").text == "js"
    assert http.get("/some/client/route").text == "<html>safe</html>"
    assert http.get("/leak").status_code == 403
    assert http.get("/%2e%2e/private.txt").status_code == 403
    response = http.get("/api/does-not-exist")
    assert response.status_code == 404 and response.json()["error"] == "not-found"


def test_spa_without_build(fixture_logs, tmp_path):
    http = make_client(fixture_logs, dist_dir=tmp_path / "no-dist")
    assert "not built" in http.get("/").text


def test_entrypoint_binds_to_loopback(monkeypatch):
    import uvicorn

    from studio import backend

    seen = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: seen.append(kwargs))
    backend.run(5000)
    assert seen[0]["host"] == "127.0.0.1"


# ---------------------------------------------------------------- PID ownership


def test_forged_pid_is_never_signalled(mut_api):
    http, root = mut_api
    pid_file = root / "healthy" / "run-0" / "pid"
    pid_file.write_text(str(os.getpid()))
    for url in ("/api/runs/healthy/run-0/stop", "/api/experiments/healthy/stop"):
        response = http.post(url)
        assert response.status_code == 403 and response.json()["error"] == "pid-unverifiable"
    assert pid_file.read_text() == str(os.getpid())


def test_symlinked_pid_file_cannot_be_used(mut_api, tmp_path):
    http, root = mut_api
    external = tmp_path / "outside-pid"
    external.write_text(str(os.getpid()))
    (root / "healthy" / "run-0" / "pid").symlink_to(external)
    assert http.post("/api/runs/healthy/run-0/stop").status_code == 403
    assert external.read_text() == str(os.getpid())


def test_stale_pid_is_removed_without_signalling_on_restart(mut_api, popen):
    http, root = mut_api
    rundir = root / "healthy" / "run-1"
    (rundir / "pid").write_text("999999999")
    for name in ("train.csv", "training_data.csv"):
        (rundir / name).unlink()
    (rundir / "test.csv").write_text((rundir / "test.csv").read_text().splitlines()[0] + "\n0,0.5,10,1.0,1700000000.0,1000\n")
    response = http.post("/api/runs/healthy/run-1/restart", json={"device": "cpu"})
    assert response.status_code == 202, response.text
    command = popen.call_args.args[0]
    assert "--seed=1" in command and "--n-runs=1" in command and "--device=cpu" in command
    assert not (rundir / "pid").exists()


class FakeLauncher:
    pid = 999

    def __init__(self, root, signals):
        self.root, self.signals = root, signals

    def cmdline(self):
        return ["python", "scripts/start_run.py", str(self.root / "healthy")]

    def send_signal(self, sig):
        self.signals.append((self.pid, sig))


def fake_process(root, signals, launcher_parent: bool):
    class FakeProcess:
        pid = 123456

        def __init__(self, pid):
            if pid != self.pid:
                raise psutil.NoSuchProcess(pid)

        def status(self):
            return psutil.STATUS_SLEEPING

        def parents(self):
            return [FakeLauncher(root, signals)] if launcher_parent else []

        def cmdline(self):
            if launcher_parent:
                return ["python", "-c", "from multiprocessing.spawn import spawn_main"]
            return ["python", "scripts/start_run.py", str(root / "healthy")]

        def send_signal(self, sig):
            signals.append((self.pid, sig))

    return FakeProcess


def test_verified_run_stop_signals_only_its_launcher_tree(mut_api, monkeypatch):
    http, root = mut_api
    pid_file = root / "healthy" / "run-0" / "pid"
    pid_file.write_text("123456")
    signals = []
    monkeypatch.setattr(security.psutil, "Process", fake_process(root, signals, launcher_parent=False))
    assert http.get("/api/experiments/healthy").json()["running_runs"] == 1
    assert http.post("/api/runs/healthy/run-0/stop").status_code == 204
    assert signals == [(123456, signal.SIGINT)]
    assert not pid_file.exists()


def test_stop_experiment_signals_verified_launcher(mut_api, monkeypatch):
    http, root = mut_api
    (root / "healthy" / "run-0" / "pid").write_text("123456")
    signals = []
    monkeypatch.setattr(security.psutil, "Process", fake_process(root, signals, launcher_parent=True))
    monkeypatch.setattr(runs.time, "sleep", lambda seconds: None)
    assert http.post("/api/experiments/healthy/stop").status_code == 204
    assert {pid for pid, _ in signals} == {123456, 999}


def test_real_running_process_is_stopped(mut_api):
    http, _ = mut_api
    detail = http.get("/api/experiments/running").json()
    pid = detail["runs"][0]["pid"]
    assert detail["status"] == "RUNNING" and pid is not None
    assert http.post("/api/runs/running/run-0/stop").status_code == 204
    deadline = time.time() + 5
    while psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE and time.time() < deadline:
        time.sleep(0.05)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert http.get("/api/experiments/running").json()["runs"][0]["status"] != "RUNNING"


def test_run_paths_come_from_the_directory_not_the_json(mut_api, tmp_path, popen):
    """Ported from test_ui_run_paths: a run.json pointing elsewhere never redirects actions."""
    http, root = mut_api
    run_json = root / "healthy" / "run-1" / "run.json"
    run_json.write_text(run_json.read_text().replace(str(root / "healthy" / "run-1"), str(tmp_path / "outside" / "run-1")))
    detail = http.get("/api/experiments/healthy").json()
    assert "logdir-mismatch" in {i["code"] for i in detail["issues"]}
    assert launcher.settings.START_RUN_SCRIPT.is_file()
