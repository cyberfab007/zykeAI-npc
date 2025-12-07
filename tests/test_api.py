import os
import json

import pytest

os.environ["REQUIRE_API_TOKEN"] = "false"
os.environ["ALLOW_CUSTOM_ADAPTER_PATH"] = "true"  # allow stub path in tests

from deployment.app import app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Stub generate_npc_response to avoid heavy model load
    def fake_generate(**kwargs):
        return {"say": "hi", "action": "idle", "emotion": "neutral"}

    monkeypatch.setattr("deployment.app.generate_npc_response", lambda **kwargs: fake_generate(**kwargs))
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_generate_single(client):
    payload = {
        "persona": "tester",
        "context": "unit test",
        "state": "ok",
        "player_input": "hello",
        "adapter_name": None,
    }
    resp = client.post("/generate", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "result" in data
    assert data["result"]["say"] == "hi"


def test_generate_batch(client):
    payload = {
        "requests": [
            {"persona": "p1", "context": "c1", "state": "s1", "player_input": "hi"},
            {"persona": "p2", "context": "c2", "state": "s2", "player_input": "hi"},
        ]
    }
    resp = client.post("/generate", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data and len(data["results"]) == 2
