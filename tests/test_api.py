import os
import json

import pytest

os.environ["REQUIRE_API_TOKEN"] = "false"
os.environ["ALLOW_CUSTOM_ADAPTER_PATH"] = "true"  # allow stub path in tests

from deployment.app import app  # noqa: E402
import deployment.app as deployment_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Stub generate_npc_response to avoid heavy model load
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"say": "hi", "action": "idle", "emotion": "neutral"}

    monkeypatch.setattr("deployment.app.generate_npc_response", fake_generate)
    c = app.test_client()
    c.captured_generate_kwargs = captured
    return c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_worker_contract_endpoints(client):
    status_resp = client.get("/status")
    assert status_resp.status_code == 200
    assert status_resp.get_json()["worker"]["bind_host"] == "127.0.0.1"

    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    assert "local_models" in models_resp.get_json()

    adapters_resp = client.get("/adapters")
    assert adapters_resp.status_code == 200
    assert "npc_core_pythia_410m_v1" in adapters_resp.get_json()["adapters"]

    training_resp = client.get("/training/status")
    assert training_resp.status_code == 200
    assert training_resp.get_json()["status"] == "idle"

    training_start_resp = client.post("/training/start", json={"job_id": "job-1"})
    assert training_start_resp.status_code == 202
    assert training_start_resp.get_json()["status"] == "accepted"

    training_stop_resp = client.post("/training/stop", json={"job_id": "job-1"})
    assert training_stop_resp.status_code == 200

    training_export_resp = client.post("/training/export-delta", json={"job_id": "job-1"})
    assert training_export_resp.status_code == 200
    assert training_export_resp.get_json()["status"] == "not_available"

    memory_resp = client.get("/memory/status")
    assert memory_resp.status_code == 200
    assert "status" in memory_resp.get_json()

    memory_query_resp = client.post("/memory/query", json={"query": "hello"})
    assert memory_query_resp.status_code == 200
    assert memory_query_resp.get_json()["entries"] == []

    memory_update_resp = client.post("/memory/update", json={"entry": "hello"})
    assert memory_update_resp.status_code == 202


def test_generate_single(client):
    payload = {
        "persona": "tester",
        "context": "unit test",
        "state": "ok",
        "player_input": "hello",
        "adapter_name": "npc_core_pythia_410m_v1",
        "adapter_version": 123,
        "enable_tools": True,
        "npc_type": "dog_guard",
    }
    resp = client.post("/generate", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "result" in data
    assert data["result"]["say"] == "hi"
    assert client.captured_generate_kwargs.get("cache_tag") == "npc_core_pythia_410m_v1:123"
    assert client.captured_generate_kwargs.get("enable_tools") is True
    assert client.captured_generate_kwargs.get("npc_type") == "dog_guard"


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


def test_generate_missing_field_validation(client):
    payload = {"persona": "p", "context": "c", "state": "s"}
    resp = client.post("/generate", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 500
    assert "Missing fields: player_input" in resp.get_json()["error"]


def test_concurrency_release_on_batch_error(client, monkeypatch):
    def fake_handle(body):
        raise ValueError("batch exploded")

    monkeypatch.setattr(deployment_app, "_handle_single_request", fake_handle)
    deployment_app.concurrency_count = 0
    payload = {"requests": [{"persona": "p", "context": "c", "state": "s", "player_input": "hi"}]}
    resp = client.post("/generate", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 500
    assert deployment_app.concurrency_count == 0
