"""Read-only experiment and series routes on the fixture logs."""

import pytest
from conftest import N_STEPS, SCENARIOS, healthy_raw, make_client, write_experiment, write_run

SUMMARY_KEYS = {"id", "name", "algo", "env", "created", "n_steps", "status", "progress", "health", "issue_counts", "n_runs", "running_runs"}
RUN_KEYS = {"id", "dirname", "seed", "status", "progress", "latest_step", "pid", "config", "issues"}
ISSUE_KEYS = {"level", "code", "message", "scope", "path", "detail"}


def test_list_and_filters(api):
    summaries = api.get("/api/experiments").json()
    assert sorted(s["id"] for s in summaries) == sorted(SCENARIOS)
    assert all(set(s) == SUMMARY_KEYS for s in summaries)
    assert [s["id"] for s in api.get("/api/experiments", params={"status": "RUNNING"}).json()] == ["running"]
    assert {s["id"] for s in api.get("/api/experiments", params={"health": "error"}).json()} == {"corrupt-run"}
    assert {s["id"] for s in api.get("/api/experiments", params={"health": "ok,error", "q": "corrupt"}).json()} == {"corrupt-run"}
    assert api.get("/api/experiments", params={"algo": "PPO,IPPO"}).json() == []
    assert {s["id"] for s in api.get("/api/experiments", params={"q": "mixer=qmixerv1"}).json()} == {"unknown-class"}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_detail_works_for_every_fixture(api, scenario):
    response = api.get(f"/api/experiments/{scenario}")
    assert response.status_code == 200
    detail = response.json()
    assert set(detail) == SUMMARY_KEYS | {"raw", "issues", "capabilities", "runs", "params"}
    assert set(detail["capabilities"]) == {"metrics", "params", "replay", "launch"}
    assert all(set(r) == RUN_KEYS for r in detail["runs"])
    assert all(set(i) == ISSUE_KEYS for i in detail["issues"])
    seeds = [r["seed"] for r in detail["runs"]]
    assert seeds == sorted(seeds)


def test_unknown_experiment_is_404_with_error_body(api):
    for url in ("/api/experiments/nope", "/api/experiments/nope/catalog", "/api/experiments/nope/test-steps", "/api/runs/nope/run-0/stop"):
        response = api.get(url) if "stop" not in url else api.post(url)
        assert response.status_code == 404, url
        assert set(response.json()) == {"error", "message"}
    assert api.get("/api/experiments/healthy/run-9/whatever").status_code == 404
    assert api.post("/api/runs/healthy/run-9/stop").json()["error"] == "unknown-run"


def test_catalog_preview_steps_episodes_params(api):
    catalog = api.get("/api/experiments/healthy/catalog").json()
    assert catalog["default_metric"] == {"table": "test", "metric": "score-0"}
    assert set(catalog["tables"]) == {"test", "train", "training_data"}
    preview = api.get("/api/experiments/healthy/preview", params={"points": 8}).json()
    assert preview["metric"] == {"table": "test", "metric": "score-0"} and len(preview["result"]["x"]) <= 8
    assert api.get("/api/experiments/healthy/preview", params={"points": 0}).status_code == 400
    assert api.get("/api/experiments/healthy/test-steps").json() == list(range(0, N_STEPS + 1, 1000))
    episodes = api.get("/api/experiments/healthy/episodes", params={"step": 1000}).json()
    assert len(episodes) == 6
    assert set(episodes[0]) == {"run", "seed", "test", "step", "metrics", "has_actions"}
    missing = api.get("/api/experiments/healthy/episodes")
    assert missing.status_code == 400 and missing.json()["error"] == "invalid-request"
    params = api.get("/api/params", params={"ids": "healthy,nope,../x"}).json()
    assert list(params) == ["healthy"]
    assert {"path", "key", "depth", "kind", "value", "cls", "curve"} == set(params["healthy"][0])


def test_series_partial_success(api):
    queries = [
        {"experiment": "healthy", "table": "test", "metric": "score-0"},
        {"experiment": "healthy", "table": "train", "metric": "score-0", "center": "none", "x": "wall_time"},
        {"experiment": "nope", "table": "test", "metric": "score-0"},
        {"experiment": "healthy", "table": "test", "metric": "score-0", "band": "wide"},
        {"experiment": "missing-table", "table": "test", "metric": "score-0"},
    ]
    out = api.post("/api/series", json={"queries": queries}).json()
    assert [o["ok"] for o in out] == [True, True, False, False, True]
    assert out[1]["result"]["center"] is None and len(out[1]["result"]["runs"]) == 3
    assert out[2]["issue"]["code"] == "unknown-experiment"
    assert out[4]["result"]["missing_runs"] == ["missing-table/run-2"]


@pytest.mark.parametrize("body", [None, {}, {"queries": []}, {"queries": "x"}, {"queries": [{}] * 201}])
def test_series_rejects_bad_batches(api, body):
    response = api.post("/api/series", json=body)
    assert response.status_code == 400 and response.json()["error"] == "invalid-request"


def test_health_endpoint_computes_capabilities(api):
    out = api.post("/api/experiments/unknown-class/health").json()
    assert out["capabilities"]["launch"] is False and out["capabilities"]["replay"] is False
    issue = next(i for i in out["issues"] if i["code"] == "deserialize-failed")
    assert issue["path"] == "trainer.mixer"
    assert api.get("/api/experiments/unknown-class").json()["capabilities"]["launch"] is False
    assert api.post("/api/experiments/healthy/health").json()["capabilities"] == {
        "metrics": True,
        "params": "full",
        "replay": True,
        "launch": True,
    }


def test_nested_experiment_ids(make_logs):
    exp = make_logs / "sweeps" / "a" / "exp-3"
    write_experiment(exp, healthy_raw(exp))
    write_run(exp, 0)
    http = make_client(make_logs)
    assert http.get("/api/experiments/sweeps/a/exp-3").json()["id"] == "sweeps/a/exp-3"
    assert http.get("/api/experiments/sweeps/a/exp-3/catalog").status_code == 200
    assert http.get("/api/experiments/sweeps%2Fa%2Fexp-3/test-steps").status_code == 200
    assert http.get("/api/experiments/sweeps/a").status_code == 404
