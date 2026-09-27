"""Smoke tests for the MARL Studio application skeleton."""

from fastapi.testclient import TestClient

from studio.backend.app import app


def test_health_endpoint():
    client = TestClient(app, base_url="http://localhost:5000")
    assert client.get("/api/health").json() == {"ok": True}


def test_unknown_api_route_is_404():
    client = TestClient(app, base_url="http://localhost:5000")
    assert client.get("/api/does-not-exist").status_code == 404
