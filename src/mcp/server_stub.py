"""
Minimal MCP-like tool server stub for local development.

This is NOT a full MCP implementation; it's a small HTTP service that exposes:
- GET /health
- GET /tools
- POST /call

Replace handlers with real game/metaverse state.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request

from src.mcp.registry import allowed_tools, load_tool_manifest, parse_tools


app = Flask(__name__)

REQUIRE_TOKEN = os.getenv("MCP_REQUIRE_TOKEN", "false").lower() == "true"
API_TOKEN = os.getenv("MCP_API_TOKEN")
MANIFEST_PATH = os.getenv("MCP_TOOL_MANIFEST", "data/mcp/tools.json")


def _auth_ok() -> bool:
    if not REQUIRE_TOKEN:
        return True
    if not API_TOKEN:
        return False
    return request.headers.get("Authorization", "") == f"Bearer {API_TOKEN}"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/tools", methods=["GET"])
def tools():
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    manifest = load_tool_manifest(MANIFEST_PATH)
    tools_list = []
    for t in parse_tools(manifest):
        tools_list.append(
            {
                "name": t.name,
                "description": t.description,
                "server": t.server,
                "allowed_npc_types": t.allowed_npc_types,
                "allowed_audiences": t.allowed_audiences,
                "args_schema": t.args_schema,
            }
        )
    return jsonify({"tools": tools_list}), 200


@app.route("/call", methods=["POST"])
def call():
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    tool_name = payload.get("tool")
    args = payload.get("args") or {}
    if not isinstance(tool_name, str) or not tool_name:
        return jsonify({"error": "tool required"}), 400
    if not isinstance(args, dict):
        return jsonify({"error": "args must be an object"}), 400

    # For now, return deterministic stub outputs.
    if tool_name == "game.get_nearby_entities":
        radius = float(args.get("radius_m", 25))
        return jsonify(
            {
                "ok": True,
                "tool": tool_name,
                "result": {
                    "radius_m": radius,
                    "entities": [
                        {"id": "enemy_001", "type": "enemy", "distance_m": min(10.0, radius), "threat": 0.7},
                        {"id": "npc_friend_01", "type": "friendly", "distance_m": min(15.0, radius), "threat": 0.0},
                    ],
                },
            }
        )
    if tool_name == "game.get_player_status":
        pid = str(args.get("player_id", "player"))
        return jsonify({"ok": True, "tool": tool_name, "result": {"player_id": pid, "hp": 100, "is_under_attack": False}})
    if tool_name == "game.command":
        return jsonify({"ok": True, "tool": tool_name, "result": {"accepted": True, "echo": args}})

    return jsonify({"ok": False, "tool": tool_name, "error": "unknown_tool"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("MCP_PORT", "7000")))

