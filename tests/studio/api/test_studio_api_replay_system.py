"""Replay (mocked marl run) and system routes."""

import numpy as np
import orjson
import pytest
from conftest import make_client

import marl
from marl.utils.gpu import GPU
from studio.backend.services import system


def test_replay_requires_replay_capability(api):
    response = api.get("/api/runs/unknown-class/run-0/replay", params={"step": 1000, "test": 0})
    assert response.status_code == 409
    assert response.json()["error"] == "not-replayable" and response.json()["issue"]["code"] == "deserialize-failed"
    assert api.get("/api/runs/healthy/run-0/replay", params={"step": 1000}).status_code == 400


def test_replay_serializes_like_the_old_route(api, fixture_logs, monkeypatch):
    from marl.models.run import Run

    calls = []

    class FakeRun:
        rundir = "logs/elsewhere/run-0"

        def replay_episode(self, time_step, test_num, only_saved_actions):
            calls.append((self.rundir, time_step, test_num, only_saved_actions))
            return {"frames": np.zeros((2, 2), dtype=np.uint8), "path": fixture_logs / "healthy" / "run-0", "tags": {"a"}}

    monkeypatch.setattr(Run, "load", classmethod(lambda cls, path: FakeRun()))
    response = api.get("/api/runs/healthy/run-0/replay", params={"step": 1000, "test": 1, "only_saved_actions": True})
    assert response.status_code == 200, response.text
    expected = orjson.dumps(
        {"frames": np.zeros((2, 2), dtype=np.uint8), "path": fixture_logs / "healthy" / "run-0", "tags": {"a"}},
        option=orjson.OPT_SERIALIZE_NUMPY,
        default=marl.utils.default_serialization,
    )
    assert response.content == expected
    assert calls == [(str(fixture_logs / "healthy" / "run-0"), 1000, 1, True)]


def test_replay_without_saved_data_is_404(api, monkeypatch):
    from marl.models.run import Run

    class FakeRun:
        rundir = ""

        def replay_episode(self, *args, **kwargs):
            raise FileNotFoundError("Could not find any data to replay the episode")

    monkeypatch.setattr(Run, "load", classmethod(lambda cls, path: FakeRun()))
    response = api.get("/api/runs/healthy/run-0/replay", params={"step": 3, "test": 0})
    assert response.status_code == 404 and response.json()["error"] == "replay-unavailable"


@pytest.fixture
def fake_gpus(monkeypatch):
    monkeypatch.setattr(system, "list_gpus", lambda: [GPU(0, 1000, 250, 750, 50)])


def test_system_specs(api, fake_gpus):
    specs = api.get("/api/system/specs").json()
    assert 0 <= specs["cpu"] <= 100 and 0 <= specs["ram"] <= 100
    assert specs["gpus"] == [
        {"index": 0, "total_memory": 1000, "used_memory": 250, "free_memory": 750, "memory_usage": 0.25, "utilization": 0.5}
    ]


def test_system_websocket(fixture_logs, fake_gpus):
    http = make_client(fixture_logs, system_interval=0.01)
    with http.websocket_connect("/api/system/ws", headers={"Host": "localhost:5000"}) as ws:
        first = orjson.loads(ws.receive_bytes())
        second = orjson.loads(ws.receive_bytes())
    assert set(first) == set(second) == {"cpu", "ram", "gpus"}
    assert first["gpus"][0]["memory_usage"] == 0.25


def test_system_websocket_rejects_foreign_origins(api, fake_gpus):
    from starlette.websockets import WebSocketDisconnect

    for headers in ({"Host": "localhost:5000", "Origin": "https://evil.example"}, {"Host": "attacker.example"}):
        with pytest.raises(WebSocketDisconnect), api.websocket_connect("/api/system/ws", headers=headers) as ws:
            ws.receive_bytes()
