from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

from src.mcp.registry import MCPServerConfig, MCPTool, load_tool_manifest, parse_servers


class MCPClientError(RuntimeError):
    pass


def _auth_header(server: MCPServerConfig) -> Dict[str, str]:
    if server.auth_type == "bearer_env" and server.auth_env_var:
        tok = os.getenv(server.auth_env_var)
        if tok:
            return {"Authorization": f"Bearer {tok}"}
    return {}


def call_tool(
    tool: MCPTool,
    args: Dict[str, Any],
    *,
    manifest_path: str = "data/mcp/tools.json",
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    manifest = load_tool_manifest(manifest_path)
    servers = parse_servers(manifest)
    if tool.server not in servers:
        raise MCPClientError(f"Unknown MCP server '{tool.server}' for tool '{tool.name}'")
    server = servers[tool.server]
    url = f"{server.base_url}/call"
    payload = {"tool": tool.name, "args": args}
    try:
        resp = requests.post(url, json=payload, headers=_auth_header(server), timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise MCPClientError(f"MCP call failed: {tool.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise MCPClientError("MCP server returned non-object JSON")
    return data

