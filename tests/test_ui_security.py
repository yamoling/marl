"""Security regression tests for the local UI backend (no real logs are modified)."""

import os
import shutil
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from ui import backend
from ui.backend import routes
from ui.backend.routes import experiments


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "logs"
    root.mkdir()
    (root / "demo").mkdir()
    (root / "demo" / "run").mkdir()
    monkeypatch.setattr(routes.state, "logdir", str(root))
    return TestClient(routes.app, base_url="http://localhost:5000"), root


def test_dist_is_independent_of_cwd_and_encoded_traversal_is_forbidden(client, tmp_path, monkeypatch):
    http, _ = client
    monkeypatch.chdir(tmp_path)
    assert http.get("/").status_code == 200
    assert http.get("/index.html").status_code == 200
    assert http.get("/%2e%2e/%2e%2e/backend/routes/__init__.py").status_code == 403
    assert http.get("/api/does-not-exist").status_code == 404


def test_spa_does_not_follow_dist_symlinks(client, tmp_path):
    http, _ = client
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    # Avoid writing into the repository: use a temporary distribution tree.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>safe</html>")
    (dist / "leak").symlink_to(secret)
    original = routes.dist_dir
    routes.dist_dir = dist
    try:
        assert http.get("/leak").status_code == 403
    finally:
        routes.dist_dir = original


@pytest.mark.parametrize(
    "route,method",
    [
        ("/experiment/load/{path}", "post"),
        ("/experiment/delete/{path}", "delete"),
        ("/experiment/{path}", "get"),
        ("/experiment/is_running/{path}", "get"),
        ("/experiment/replay/0/0/true/{path}", "get"),
        ("/experiment/image/0/{path}", "get"),
        ("/runs/get/{path}", "get"),
        ("/runs/start/{path}", "post"),
        ("/runs/stop/{path}", "post"),
        ("/results/load/{path}?granularity=1", "get"),
        ("/results/test/0/{path}", "get"),
        ("/runner/new/{path}", "post"),
    ],
)
def test_paths_outside_root_rejected_before_state(client, tmp_path, monkeypatch, route, method):
    http, root = client
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before calling state")

    for name in ("get_experiment", "load_experiment", "unload_experiment", "replay_episode", "start_run", "stop_run", "new_runs"):
        monkeypatch.setattr(routes.state, name, forbidden)
    path = quote(str(root / "escape"), safe="/")
    response = (
        getattr(http, method)(route.format(path=path), json={"nRuns": 1, "nTests": 1, "seed": 1} if method == "post" else None)
        if method == "post"
        else getattr(http, method)(route.format(path=path))
    )
    assert response.status_code == 403


def test_missing_and_root_paths_have_honest_statuses(client):
    http, root = client
    assert http.get(f"/runs/get/{root / 'missing'}").status_code == 404
    assert http.get(f"/runs/get/{root}").status_code == 400
    assert http.get(f"/runs/get/{root / 'demo' / '..' / '..' / 'outside'}").status_code == 403


def test_valid_paths_keep_client_path_style(client, monkeypatch):
    http, root = client
    seen = []
    monkeypatch.setattr(routes.state, "load_experiment", lambda path: seen.append(path))
    assert http.post(f"/experiment/load/{root / 'demo'}").status_code == 204
    assert seen == [str(root / "demo")]
    assert http.post("/experiment/load/logs/demo").status_code == 204
    assert seen == [str(root / "demo"), str(root / "demo")]
    monkeypatch.setattr(routes.state, "new_runs", lambda path, *a, **kw: seen.append(path))
    assert http.post("/runner/new/logs/demo", json={"nRuns": 1, "nTests": 1, "seed": 1}).status_code == 200
    assert seen[-1] == str(root / "demo")


def test_rename_rejects_source_and_destination_escapes(client, tmp_path, monkeypatch):
    http, root = client
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: (_ for _ in ()).throw(AssertionError("must not load")))
    for payload in (
        {"logdir": str(outside), "newLogdir": str(root / "renamed")},
        {"logdir": str(root / "demo"), "newLogdir": str(outside / "new")},
        {"logdir": str(root / "demo"), "newLogdir": str(root / "escape" / "new")},
    ):
        assert http.post("/experiment/rename", json=payload).status_code == 403
    assert http.post("/experiment/rename", json={"logdir": str(root / "demo")}).status_code == 400
    assert (
        http.post("/experiment/rename", json={"logdir": str(root / "demo"), "newLogdir": str(root / "demo" / "nested")}).status_code == 400
    )
    assert (
        http.post("/experiment/rename", json={"logdir": str(root / "demo"), "newLogdir": str(root / "missing" / "nested")}).status_code
        == 404
    )


def test_rename_checks_serialized_paths_before_moving(client, tmp_path, monkeypatch):
    http, root = client
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = []
    exp = SimpleNamespace(logdir=str(outside), runs=[], move=lambda path: moved.append(path))
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    payload = {"logdir": str(root / "demo"), "newLogdir": str(root / "renamed")}
    assert http.post("/experiment/rename", json=payload).status_code == 403
    exp.logdir = str(root / "demo")
    exp.runs = [SimpleNamespace(rundir=str(outside / "run"))]
    assert http.post("/experiment/rename", json=payload).status_code == 403
    assert moved == []
    exp.runs = [SimpleNamespace(rundir=str(root / "demo" / "run"))]
    monkeypatch.setattr(routes.state, "unload_experiment", lambda path: None)
    monkeypatch.setattr(routes.state, "load_experiment", lambda path: None)
    assert http.post("/experiment/rename", json=payload).status_code == 204
    assert moved == [root / "renamed"]


def test_relative_serialized_logdir_still_works(client, tmp_path, monkeypatch):
    http, root = client
    other_cwd = tmp_path / "src"
    (other_cwd / "logs" / "demo").mkdir(parents=True)
    (other_cwd / "logs" / "demo" / "untouched").write_text("keep")
    monkeypatch.chdir(other_cwd)
    exp = SimpleNamespace(logdir="logs/demo", runs=[])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    monkeypatch.setattr(routes.state, "unload_experiment", lambda path: None)
    assert http.delete(f"/experiment/delete/{root / 'demo'}").status_code == 204
    assert not (root / "demo").exists()
    assert (other_cwd / "logs" / "demo" / "untouched").read_text() == "keep"


def test_rename_from_alternate_cwd_uses_only_the_logs_root(client, tmp_path, monkeypatch):
    http, root = client
    other_cwd = tmp_path / "src"
    (other_cwd / "logs" / "demo").mkdir(parents=True)
    (other_cwd / "logs" / "demo" / "untouched").write_text("keep")
    monkeypatch.chdir(other_cwd)
    exp = SimpleNamespace(logdir="logs/demo", runs=[])
    exp.move = lambda target: shutil.move(exp.logdir, target)
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    monkeypatch.setattr(routes.state, "unload_experiment", lambda path: None)
    monkeypatch.setattr(routes.state, "load_experiment", lambda path: None)
    payload = {"logdir": "logs/demo", "newLogdir": "logs/renamed"}
    assert http.post("/experiment/rename", json=payload).status_code == 204
    assert (root / "renamed" / "run").is_dir()
    assert not (root / "demo").exists()
    assert (other_cwd / "logs" / "demo" / "untouched").read_text() == "keep"


def test_delete_never_follows_serialized_metadata_or_unsafe_fallback(client, tmp_path, monkeypatch):
    http, root = client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("keep")
    monkeypatch.setattr(
        routes.state, "get_experiment", lambda path: SimpleNamespace(logdir=str(outside), delete=lambda: pytest.fail("unsafe delete"))
    )
    assert http.delete(f"/experiment/delete/{root / 'demo'}").status_code == 403
    assert (outside / "sentinel").exists()
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: (_ for _ in ()).throw(AttributeError("old format")))
    assert http.delete(f"/experiment/delete/{outside}").status_code == 403
    assert (outside / "sentinel").exists()
    assert http.delete(f"/experiment/delete/{root / 'demo'}").status_code == 204
    assert not (root / "demo").exists()


def test_cross_origin_mutations_are_blocked_but_local_dev_origins_work(client, monkeypatch):
    http, root = client
    seen = []
    monkeypatch.setattr(routes.state, "load_experiment", lambda path: seen.append(path))
    url = f"/experiment/load/{root / 'demo'}"
    assert http.post(url, headers={"Origin": "https://evil.example"}).status_code == 403
    assert http.post(url, headers={"Origin": "http://localhost.evil.example"}).status_code == 403
    assert http.post(url, headers={"Origin": "http://localhost:bad"}).status_code == 403
    assert http.post(url, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert not seen
    assert http.post(url, headers={"Origin": "http://localhost:5173"}).status_code == 204
    assert http.post(url, headers={"Origin": "http://127.0.0.1:5173"}).status_code == 204
    assert http.post(url, headers={"Origin": "http://localhost:5174"}).status_code == 204
    assert len(seen) == 3
    assert (
        http.options(url, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}).headers.get(
            "access-control-allow-origin"
        )
        is None
    )


def test_alternate_cwd_replay_normalizes_serialized_run_paths(client, tmp_path, monkeypatch):
    http, root = client
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / "src")
    full = SimpleNamespace(rundir="logs/demo/run", replay_episode=lambda *a, **kw: {"ok": True})
    run = SimpleNamespace(rundir="logs/demo/run", to_full=lambda: full)
    exp = SimpleNamespace(logdir="logs/demo", runs=[run])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path, full=False: exp)
    response = http.get("/experiment/replay/0/0/true/logs/demo/run")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert full.rundir == str(root / "demo" / "run")


def test_external_serialized_run_cannot_be_stopped_or_replayed(client, tmp_path, monkeypatch):
    http, root = client
    outside = tmp_path / "outside"
    outside.mkdir()
    run = SimpleNamespace(rundir=str(outside), to_full=lambda: pytest.fail("unsafe replay"))
    exp = SimpleNamespace(logdir="logs/demo", runs=[run])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path, full=False: exp)
    assert http.post(f"/runs/stop/{root / 'demo' / 'run'}").status_code == 403
    assert http.get(f"/experiment/replay/0/0/true/{root / 'demo' / 'run'}").status_code == 403
    assert http.post(f"/experiment/stop-runs/{root / 'demo'}").status_code == 403
    assert http.get(f"/runs/get/{root / 'demo'}").status_code == 403
    assert http.get(f"/results/load/{root / 'demo'}?granularity=1").status_code == 403
    assert http.get(f"/results/test/0/{root / 'demo'}").status_code == 403


def test_alternate_cwd_start_run_uses_validated_metadata(client, tmp_path, monkeypatch):
    http, root = client
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path / "src")
    run = SimpleNamespace(
        rundir="logs/demo/run",
        is_complete=False,
        seed=7,
        test_interval=5,
        save_weights=False,
        save_actions=True,
    )
    exp = SimpleNamespace(logdir="logs/demo", runs=[run])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    started = []
    monkeypatch.setattr(routes.state, "new_runs", lambda path, **kwargs: started.append((path, kwargs)))
    assert http.post("/runs/start/logs/demo/run", json={"device": "cpu"}).status_code == 204
    assert started[0][0] == str(root / "demo")
    assert started[0][1]["seed"] == 7
    assert started[0][1]["device"] == "cpu"


def test_stale_pid_is_removed_without_signalling(client, monkeypatch):
    http, root = client
    pid_file = root / "demo" / "run" / "pid"
    pid_file.write_text("999999999")
    run = SimpleNamespace(
        rundir="logs/demo/run",
        is_complete=False,
        seed=7,
        test_interval=5,
        save_weights=False,
        save_actions=True,
    )
    exp = SimpleNamespace(logdir="logs/demo", runs=[run])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    launched = []
    monkeypatch.setattr(routes.state, "new_runs", lambda path, **kwargs: launched.append(path))
    assert http.post("/runs/start/logs/demo/run", json={}).status_code == 204
    assert launched == [str(root / "demo")]
    assert not pid_file.exists()


def test_forged_pid_is_never_signalled(client, monkeypatch):
    http, root = client
    pid_file = root / "demo" / "run" / "pid"
    pid_file.write_text(str(os.getpid()))
    run = SimpleNamespace(rundir="logs/demo/run")
    exp = SimpleNamespace(logdir="logs/demo", runs=[run])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    assert http.post(f"/runs/stop/{root / 'demo' / 'run'}").status_code == 403
    assert http.post(f"/experiment/stop-runs/{root / 'demo'}").status_code == 403
    assert pid_file.read_text() == str(os.getpid())


def test_symlinked_pid_file_cannot_be_used_to_stop_run(client, tmp_path, monkeypatch):
    http, root = client
    external_pid = tmp_path / "outside-pid"
    external_pid.write_text(str(os.getpid()))
    (root / "demo" / "run" / "pid").symlink_to(external_pid)
    exp = SimpleNamespace(logdir="logs/demo", runs=[SimpleNamespace(rundir="logs/demo/run")])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    assert http.post(f"/runs/stop/{root / 'demo' / 'run'}").status_code == 403
    assert external_pid.read_text() == str(os.getpid())


def test_verified_run_stop_signals_only_its_launcher_tree(client, monkeypatch):
    http, root = client
    pid_file = root / "demo" / "run" / "pid"
    pid_file.write_text("123456")
    signals = []

    class FakeProcess:
        pid = 123456

        def __init__(self, pid):
            assert pid == self.pid

        def parents(self):
            return []

        def cmdline(self):
            return ["python", "scripts/start_run.py", str(root / "demo")]

        def send_signal(self, sig):
            signals.append(sig)

    monkeypatch.setattr(routes.psutil, "Process", FakeProcess)
    exp = SimpleNamespace(logdir="logs/demo", runs=[SimpleNamespace(rundir="logs/demo/run")])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    assert http.post(f"/runs/stop/{root / 'demo' / 'run'}").status_code == 204
    assert len(signals) == 1
    assert not pid_file.exists()


def test_stop_experiment_signals_verified_launcher(client, monkeypatch):
    http, root = client
    (root / "demo" / "run" / "pid").write_text("123456")
    signals = []

    class Launcher:
        pid = 999

        def cmdline(self):
            return ["python", "scripts/start_run.py", str(root / "demo")]

        def send_signal(self, signal):
            signals.append((self.pid, signal))

    class Worker:
        pid = 123456

        def __init__(self, pid):
            assert pid == self.pid

        def cmdline(self):
            return ["python", "-c", "from multiprocessing.spawn import spawn_main"]

        def parents(self):
            return [Launcher()]

        def send_signal(self, signal):
            signals.append((self.pid, signal))

    monkeypatch.setattr(routes.psutil, "Process", Worker)
    monkeypatch.setattr(experiments.time, "sleep", lambda seconds: None)
    exp = SimpleNamespace(logdir="logs/demo", runs=[SimpleNamespace(rundir="logs/demo/run")])
    monkeypatch.setattr(routes.state, "get_experiment", lambda path: exp)
    assert http.post(f"/experiment/stop-runs/{root / 'demo'}").status_code == 204
    assert {pid for pid, _ in signals} == {123456, 999}


def test_host_and_mutation_origin_must_be_local(client, monkeypatch):
    http, root = client
    monkeypatch.setattr(routes.state, "load_experiment", lambda path: pytest.fail("unsafe mutation"))
    url = f"/experiment/load/{root / 'demo'}"
    assert http.get("/", headers={"Host": "attacker.example"}).status_code == 403
    assert http.post(url, headers={"Host": "attacker.example"}).status_code == 403
    assert http.post(url, headers={"Host": "localhost.evil"}).status_code == 403
    assert http.post(url, headers={"Origin": "http://localhost:9000"}).status_code == 403
    assert http.post(url, headers={"Origin": "http://localhost:5180"}).status_code == 403
    assert http.post(url, headers={"Origin": "http://localhost:5173", "Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert http.options(url, headers={"Host": "attacker.example", "Origin": "http://localhost:5173"}).status_code == 403


def test_nonpositive_granularity_rejected_before_result_loading(client, monkeypatch):
    http, root = client
    monkeypatch.setattr(routes.state, "get_experiment", lambda *args, **kwargs: pytest.fail("must not load results"))
    assert http.get(f"/results/load/{root / 'demo'}?granularity=0").status_code == 422
    assert http.get(f"/results/load/{root / 'demo'}?granularity=-2").status_code == 422


def test_entrypoints_bind_to_loopback(monkeypatch):
    import uvicorn

    seen = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: seen.append(kwargs))
    backend.run(5000)
    routes.run(5000)
    assert [call["host"] for call in seen] == ["127.0.0.1", "127.0.0.1"]
