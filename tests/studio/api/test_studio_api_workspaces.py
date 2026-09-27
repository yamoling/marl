"""Workspace roots select the library without moving experiment files."""

import json
from typing import cast

from conftest import make_client
from fastapi import FastAPI


def _experiment(root, name):
    path = root / name
    path.mkdir(parents=True)
    (path / "experiment.json").write_text(json.dumps({"name": name}))


def test_workspace_lifecycle_keeps_global_library(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    _experiment(root, "alpha")
    store = tmp_path / "workspaces.json"
    with make_client(root, workspace_file=store) as http:
        initial = http.get("/api/workspaces").json()
        default = initial["selected"]
        assert initial["workspaces"] == [{"id": default, "name": "Default", "logdir": str(root.resolve())}]
        app = cast(FastAPI, http.app)
        library = app.state.library
        events = app.state.events
        created = http.post("/api/workspaces", json={"name": "Other"})
        assert created.status_code == 201
        key = created.json()["id"]
        assert created.json() == {"id": key, "name": "Other", "logdir": str(root.resolve())}
        assert http.patch(f"/api/workspaces/{key}", json={"name": "Research"}).json() == {
            "id": key,
            "name": "Research",
            "logdir": str(root.resolve()),
        }
        assert http.post(f"/api/workspaces/{key}/select").json() == {"id": key, "name": "Research", "logdir": str(root.resolve())}
        assert http.get("/api/workspaces").json()["selected"] == key
        assert http.get("/api/experiments/alpha").status_code == 200
        assert app.state.library is library
        assert events.library is library
        assert library.roots == (root.resolve(),)
        assert default != key
    assert json.loads(store.read_text()) == {
        "selected": key,
        "workspaces": {
            default: {"name": "Default", "logdir": str(root.resolve())},
            key: {"name": "Research", "logdir": str(root.resolve())},
        },
    }
    with make_client(root, workspace_file=store) as http:
        assert http.get("/api/workspaces").json() == {
            "selected": key,
            "workspaces": [
                {"id": default, "name": "Default", "logdir": str(root.resolve())},
                {"id": key, "name": "Research", "logdir": str(root.resolve())},
            ],
        }
        assert http.get("/api/experiments/alpha").status_code == 200


def test_browse_directories_on_server(tmp_path):
    """The picker lists directories, not files, and rejects invalid or relative paths. @ai-generated"""
    root = tmp_path / "logs"
    root.mkdir()
    (root / "alpha").mkdir()
    (root / "Beta").mkdir()
    (root / "note.txt").write_text("not a directory")
    with make_client(root, workspace_file=tmp_path / "workspaces.json") as http:
        expected = {
            "path": str(root.resolve()),
            "parent": str(tmp_path.resolve()),
            "directories": [
                {"name": "alpha", "path": str(root / "alpha")},
                {"name": "Beta", "path": str(root / "Beta")},
            ],
        }
        assert http.get("/api/workspaces/directories").json() == expected
        assert http.get("/api/workspaces/directories", params={"path": str(root)}).json() == expected
        assert http.get("/api/workspaces/directories", params={"path": str(root / "alpha")}).json()["directories"] == []
        for path in ("relative/path", str(root / "note.txt"), str(root / "missing")):
            assert http.get("/api/workspaces/directories", params={"path": path}).status_code == 400


def test_legacy_logdirs_are_ignored_even_if_missing_and_cli_root_wins(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    _experiment(old, "old-experiment")
    _experiment(new, "new-experiment")
    store = tmp_path / "workspaces.json"
    store.write_text(
        json.dumps(
            {
                "selected": "other",
                "workspaces": {
                    "default": {"name": "Default", "logdirs": [str(old)]},
                    "other": {"name": "Other", "logdirs": [str(tmp_path / "nonexistent")]},
                },
            }
        )
    )
    with make_client(new, workspace_file=store) as http:
        assert http.get("/api/workspaces").json() == {
            "selected": "other",
            "workspaces": [
                {"id": "default", "name": "Default", "logdir": str(new.resolve())},
                {"id": "other", "name": "Other", "logdir": str(new.resolve())},
            ],
        }
        assert http.get("/api/experiments/new-experiment").status_code == 200
        assert http.get("/api/experiments/old-experiment").status_code == 404
        assert http.post("/api/workspaces/default/select").status_code == 200
        assert http.get("/api/experiments/new-experiment").status_code == 200
        assert http.get("/api/experiments/old-experiment").status_code == 404
        assert http.post("/api/workspaces/unknown/select").status_code == 404
        assert http.post("/api/workspaces", json={"name": " "}).status_code == 400
        assert http.post("/api/workspaces/default/logdirs", json={"path": str(old)}).status_code == 405
        assert http.request("DELETE", "/api/workspaces/default/logdirs", json={"path": str(new)}).status_code == 405
    assert json.loads(store.read_text()) == {
        "selected": "default",
        "workspaces": {
            "default": {"name": "Default", "logdir": str(new.resolve())},
            "other": {"name": "Other", "logdir": str(new.resolve())},
        },
    }


def test_delete_workspace_keeps_library_when_no_workspaces_remain(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    _experiment(root, "safe")
    store = tmp_path / "workspaces.json"
    with make_client(root, workspace_file=store) as http:
        first = http.get("/api/workspaces").json()["selected"]
        other = http.post("/api/workspaces", json={"name": "Other"}).json()["id"]
        http.post(f"/api/workspaces/{other}/select")
        assert http.delete(f"/api/workspaces/{other}").json()["selected"] == first
        assert http.get("/api/experiments/safe").status_code == 200
        assert http.delete(f"/api/workspaces/{first}").json() == {"selected": None, "workspaces": []}
        assert [row["id"] for row in http.get("/api/experiments").json()] == ["safe"]
        assert http.delete(f"/api/workspaces/{first}").status_code == 404
    assert (root / "safe" / "experiment.json").exists()
    with make_client(root, workspace_file=store) as http:
        assert http.get("/api/workspaces").json() == {"selected": None, "workspaces": []}
        assert http.get("/api/experiments/safe").status_code == 200


def test_explicit_root_uses_isolated_workspace_file(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    with make_client(root) as http:
        http.post("/api/workspaces", json={"name": "Local"})
    assert (root / ".studio-workspaces.json").exists()


def test_workspace_roots_switch_and_persist(tmp_path):
    """Switching roots invalidates cached IDs and the event snapshot. @ai-generated"""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _experiment(first, "alpha")
    _experiment(first, "shared")
    _experiment(second, "beta")
    _experiment(second, "shared")
    store = tmp_path / "workspaces.json"
    with make_client(first, workspace_file=store) as http:
        app = cast(FastAPI, http.app)
        original = http.get("/api/workspaces").json()["selected"]
        assert {r["id"] for r in http.get("/api/experiments").json()} == {"alpha", "shared"}
        assert app.state.library.get("shared").path == first / "shared"
        created = http.post("/api/workspaces", json={"name": "Second", "logdir": str(second)})
        assert created.status_code == 201
        key = created.json()["id"]
        assert app.state.events._workspace_epoch == 0
        assert http.post("/api/workspaces", json={"name": "Missing", "logdir": str(tmp_path / "missing")}).status_code == 400
        assert http.post(f"/api/workspaces/{key}/select").status_code == 200
        assert app.state.events._workspace_epoch == 1
        assert {r["id"] for r in http.get("/api/experiments").json()} == {"beta", "shared"}
        assert http.get("/api/experiments/alpha").status_code == 404
        assert app.state.library.get("shared").path == second / "shared"
        assert http.patch(f"/api/workspaces/{key}/logdir", json={"logdir": str(first)}).status_code == 200
        assert {r["id"] for r in http.get("/api/experiments").json()} == {"alpha", "shared"}
        assert app.state.events._workspace_epoch == 2
        assert http.patch(f"/api/workspaces/{key}/logdir", json={"logdir": str(tmp_path / "missing")}).status_code == 400
        assert http.get("/api/workspaces").json()["workspaces"][1]["logdir"] == str(first)
        assert http.patch(f"/api/workspaces/{key}/logdir", json={"logdir": str(second)}).status_code == 200
        second.rename(tmp_path / "moved")
        assert http.post(f"/api/workspaces/{original}/select").status_code == 200
        assert http.post(f"/api/workspaces/{key}/select").status_code == 400
        assert http.delete(f"/api/workspaces/{key}").json()["selected"] == original
        assert app.state.library.root == first
    with make_client(first, workspace_file=store) as http:
        assert {r["id"] for r in http.get("/api/experiments").json()} == {"alpha", "shared"}


def test_selected_root_restored_after_restart(tmp_path):
    """The selected workspace root wins over the startup default on restart. @ai-generated"""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _experiment(first, "alpha")
    _experiment(second, "beta")
    store = tmp_path / "workspaces.json"
    with make_client(first, workspace_file=store) as http:
        key = http.post("/api/workspaces", json={"name": "Second", "logdir": str(second)}).json()["id"]
        http.post(f"/api/workspaces/{key}/select")
    with make_client(first, workspace_file=store) as http:
        assert http.get("/api/workspaces").json()["selected"] == key
        assert [r["id"] for r in http.get("/api/experiments").json()] == ["beta"]
