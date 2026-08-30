from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class MCPManifestError(ValueError):
    pass


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    base_url: str
    auth_type: str = "none"
    auth_env_var: Optional[str] = None


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    server: str
    allowed_npc_types: List[str]
    allowed_audiences: List[str]
    args_schema: Dict[str, Any]


def load_tool_manifest(manifest_path: str = "data/mcp/tools.json") -> Dict[str, Any]:
    p = Path(manifest_path)
    if not p.exists():
        raise MCPManifestError(f"MCP tool manifest not found: {manifest_path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCPManifestError(f"Invalid MCP tool manifest JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MCPManifestError("MCP tool manifest must be a JSON object")
    if "servers" not in data or "tools" not in data:
        raise MCPManifestError("MCP tool manifest must contain 'servers' and 'tools'")
    return data


def parse_servers(manifest: Dict[str, Any]) -> Dict[str, MCPServerConfig]:
    servers_raw = manifest.get("servers") or {}
    if not isinstance(servers_raw, dict):
        raise MCPManifestError("'servers' must be an object")
    out: Dict[str, MCPServerConfig] = {}
    for name, cfg in servers_raw.items():
        if not isinstance(cfg, dict):
            raise MCPManifestError(f"Server '{name}' must be an object")
        base_url = cfg.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise MCPManifestError(f"Server '{name}' missing base_url")
        auth = cfg.get("auth") or {"type": "none"}
        if not isinstance(auth, dict):
            auth = {"type": "none"}
        auth_type = str(auth.get("type") or "none")
        auth_env_var = auth.get("env_var")
        if auth_env_var is not None:
            auth_env_var = str(auth_env_var)
        out[name] = MCPServerConfig(name=name, base_url=base_url.rstrip("/"), auth_type=auth_type, auth_env_var=auth_env_var)
    return out


def parse_tools(manifest: Dict[str, Any]) -> List[MCPTool]:
    tools_raw = manifest.get("tools") or []
    if not isinstance(tools_raw, list):
        raise MCPManifestError("'tools' must be a list")
    out: List[MCPTool] = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        server = t.get("server")
        desc = t.get("description") or ""
        if not isinstance(name, str) or not name:
            raise MCPManifestError("Tool missing name")
        if not isinstance(server, str) or not server:
            raise MCPManifestError(f"Tool '{name}' missing server")
        allowed_npc_types = t.get("allowed_npc_types") or ["*"]
        allowed_audiences = t.get("allowed_audiences") or ["adult", "minor"]
        if not isinstance(allowed_npc_types, list):
            allowed_npc_types = ["*"]
        if not isinstance(allowed_audiences, list):
            allowed_audiences = ["adult", "minor"]
        args_schema = t.get("args_schema") or {"type": "object"}
        if not isinstance(args_schema, dict):
            args_schema = {"type": "object"}
        out.append(
            MCPTool(
                name=name,
                description=str(desc),
                server=server,
                allowed_npc_types=[str(x) for x in allowed_npc_types],
                allowed_audiences=[str(x) for x in allowed_audiences],
                args_schema=args_schema,
            )
        )
    return out


def allowed_tools(
    *,
    manifest_path: str = "data/mcp/tools.json",
    npc_type: str = "generic",
    audience: str = "adult",
) -> Dict[str, MCPTool]:
    manifest = load_tool_manifest(manifest_path)
    tools = parse_tools(manifest)
    out: Dict[str, MCPTool] = {}
    aud = (audience or "adult").lower()
    npc = (npc_type or "generic").lower()
    for t in tools:
        if aud not in [a.lower() for a in t.allowed_audiences]:
            continue
        allowed_types = [x.lower() for x in t.allowed_npc_types]
        if "*" not in allowed_types and npc not in allowed_types:
            continue
        out[t.name] = t
    return out

