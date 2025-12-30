import os

import pytest


def test_mcp_registry_filters(tmp_path, monkeypatch):
    # Use real manifest in repo
    from src.mcp.registry import allowed_tools

    tools_generic = allowed_tools(manifest_path="data/mcp/tools.json", npc_type="generic", audience="adult")
    assert "game.get_nearby_entities" not in tools_generic

    tools_dog = allowed_tools(manifest_path="data/mcp/tools.json", npc_type="dog_guard", audience="adult")
    assert "game.get_nearby_entities" in tools_dog
    assert "game.command" in tools_dog


def test_mcp_client_call(monkeypatch):
    from src.mcp.client import call_tool
    from src.mcp.registry import allowed_tools

    tools = allowed_tools(manifest_path="data/mcp/tools.json", npc_type="dog_guard", audience="adult")
    tool = tools["game.get_nearby_entities"]

    class DummyResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url.endswith("/call")
        assert json["tool"] == "game.get_nearby_entities"
        return DummyResp({"ok": True, "result": {"entities": []}})

    monkeypatch.setattr("requests.post", fake_post)
    out = call_tool(tool, args={"radius_m": 10}, manifest_path="data/mcp/tools.json", timeout_sec=1.0)
    assert out["ok"] is True

