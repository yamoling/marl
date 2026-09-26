"""Rename (raw JSON), delete, and their refusal while runs are active."""

import json

import orjson


def test_rename_degraded_experiment_rewrites_paths_only(mut_api):
    http, root = mut_api
    before = json.loads((root / "unknown-class" / "experiment.json").read_text())
    run_before = json.loads((root / "unknown-class" / "run-0" / "run.json").read_text())
    (root / "sweeps").mkdir()
    response = http.patch("/api/experiments/unknown-class", json={"new_id": "sweeps/renamed"})
    assert response.status_code == 200 and response.json() == {"id": "sweeps/renamed"}
    target = root / "sweeps" / "renamed"
    assert not (root / "unknown-class").exists() and target.is_dir()
    after = orjson.loads((target / "experiment.json").read_bytes())
    assert after["logdir"] == str(target)
    assert {k: v for k, v in after.items() if k != "logdir"} == {k: v for k, v in before.items() if k != "logdir"}
    assert list(after) == list(before)
    run_after = json.loads((target / "run-0" / "run.json").read_text())
    assert run_after["rundir"] == str(target / "run-0")
    assert {k: v for k, v in run_after.items() if k != "rundir"} == {k: v for k, v in run_before.items() if k != "rundir"}
    detail = http.get("/api/experiments/sweeps/renamed").json()
    assert "logdir-mismatch" not in {i["code"] for i in detail["issues"]}
    assert http.get("/api/experiments/unknown-class").status_code == 404


def test_rename_keeps_relative_path_style(mut_api):
    http, root = mut_api
    exp_file = root / "healthy" / "experiment.json"
    raw = json.loads(exp_file.read_text())
    raw["logdir"] = "logs/healthy"
    exp_file.write_text(json.dumps(raw))
    assert http.patch("/api/experiments/healthy", json={"new_id": "renamed"}).status_code == 200
    assert json.loads((root / "renamed" / "experiment.json").read_text())["logdir"] == "logs/renamed"


def test_rename_refusals(mut_api):
    http, root = mut_api
    assert http.patch("/api/experiments/healthy", json={"new_id": "light"}).json()["error"] == "destination-exists"
    assert http.patch("/api/experiments/healthy", json={"new_id": "healthy/inner"}).status_code == 400
    assert http.patch("/api/experiments/healthy", json={"new_id": 3}).status_code == 400
    response = http.patch("/api/experiments/running", json={"new_id": "elsewhere"})
    assert response.status_code == 409 and response.json()["error"] == "runs-active"
    assert (root / "running").is_dir() and not (root / "elsewhere").exists()


def test_rename_experiment_without_json(mut_api):
    http, root = mut_api
    assert http.patch("/api/experiments/no-experiment-json", json={"new_id": "recovered"}).status_code == 200
    assert (root / "recovered" / "run-0" / "run.json").is_file()
    assert json.loads((root / "recovered" / "run-0" / "run.json").read_text())["rundir"] == str(root / "recovered" / "run-0")


def test_delete(mut_api):
    http, root = mut_api
    assert http.delete("/api/experiments/unknown-class").status_code == 204
    assert not (root / "unknown-class").exists()
    assert http.get("/api/experiments/unknown-class").status_code == 404
    response = http.delete("/api/experiments/running")
    assert response.status_code == 409 and (root / "running").is_dir()


def test_stop_experiment_then_delete(mut_api):
    http, _ = mut_api
    assert http.post("/api/experiments/running/stop").status_code == 204
    assert http.get("/api/experiments/running").json()["running_runs"] == 0
    assert http.delete("/api/experiments/running").status_code == 204
