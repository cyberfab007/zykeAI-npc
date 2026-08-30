from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional

from src.inference.generator import generate_npc_response
from src.mcp.client import call_tool
from src.mcp.registry import allowed_tools

from npc.memory import NPCEpisodeMemory


def parse_args():
    p = argparse.ArgumentParser(description="Simple NPC agent loop (sense→think→act) using MCP + adapters.")
    p.add_argument("--npc-type", default="dog_guard", help="NPC type used for MCP allowlisting.")
    p.add_argument("--audience", default="adult", choices=["adult", "minor"])
    p.add_argument("--adapter-name", default=None, help="Adapter name (resolved via data/adapters/manifest.json).")
    p.add_argument("--manifest-path", default="data/adapters/manifest.json")
    p.add_argument("--base-model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--tokenizer-path", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--tool-manifest", default="data/mcp/tools.json")
    p.add_argument("--tick-sec", type=float, default=1.0)
    p.add_argument("--owner-id", default="player")
    p.add_argument("--dry-run", action="store_true", help="Do not execute game.command; only print actions.")
    return p.parse_args()


def _sense(tools: Dict[str, object], tool_manifest: str, npc_type: str, audience: str, owner_id: str) -> Dict[str, Any]:
    sensed: Dict[str, Any] = {}
    if "game.get_nearby_entities" in tools:
        sensed["nearby"] = call_tool(
            tools["game.get_nearby_entities"],  # type: ignore[index]
            args={"radius_m": 25, "max_results": 20},
            manifest_path=tool_manifest,
            timeout_sec=5.0,
        ).get("result")
    if "game.get_player_status" in tools:
        sensed["owner"] = call_tool(
            tools["game.get_player_status"],  # type: ignore[index]
            args={"player_id": owner_id},
            manifest_path=tool_manifest,
            timeout_sec=5.0,
        ).get("result")
    return sensed


def _pick_target_id(nearby: Optional[Dict[str, Any]]) -> Optional[str]:
    if not nearby:
        return None
    entities = nearby.get("entities") if isinstance(nearby, dict) else None
    if not isinstance(entities, list):
        return None
    # Pick highest threat enemy in range.
    best = None
    best_threat = -1.0
    for e in entities:
        if not isinstance(e, dict):
            continue
        if str(e.get("type")) != "enemy":
            continue
        threat = float(e.get("threat") or 0.0)
        if threat > best_threat:
            best_threat = threat
            best = str(e.get("id"))
    return best


def main():
    args = parse_args()
    tools = allowed_tools(manifest_path=args.tool_manifest, npc_type=args.npc_type, audience=args.audience)
    memory = NPCEpisodeMemory(owner_id=args.owner_id, goals=["protect owner", "stay alert"])

    persona = "A protective, smart dog companion. Loyal, alert, and decisive."
    state = "You are on patrol with your owner."

    print(f"[agent] npc_type={args.npc_type} audience={args.audience} tools={list(tools.keys())}")
    while True:
        sensed = _sense(tools, args.tool_manifest, args.npc_type, args.audience, args.owner_id)
        memory_text = memory.as_context_text()
        context = "Environment signals:\n" + json.dumps(sensed, ensure_ascii=False) + ("\n\n" + memory_text if memory_text else "")

        # “Player input” in a game loop is typically the latest event (sound, threat, command).
        player_input = "Stay close and protect me. Respond with your next action."

        result = generate_npc_response(
            base_model=args.base_model,
            tokenizer_path=args.tokenizer_path,
            adapter_name=args.adapter_name,
            manifest_path=args.manifest_path,
            persona=persona,
            context=context,
            state=state,
            player_input=player_input,
            safe_mode=(args.audience != "adult"),
            enforce_schema=True,
            enable_tools=False,  # we do explicit sensing here
            npc_type=args.npc_type,
            audience=args.audience,
        )

        say = result.get("say")
        action = result.get("action")
        emotion = result.get("emotion")
        thoughts = result.get("thoughts")
        memory.add_event(f"Decided action={action} emotion={emotion} say={say}")

        target_id = _pick_target_id(sensed.get("nearby"))
        if action in {"attack", "defend"} and target_id:
            cmd = {"action": "attack", "target_id": target_id}
        elif action in {"warn", "point"}:
            cmd = {"action": "guard", "target_id": args.owner_id}
        else:
            cmd = {"action": "follow", "target_id": args.owner_id}

        if "game.command" in tools:
            if args.dry_run:
                print(f"[agent] say={say!r} action={action} -> game.command {cmd}")
            else:
                resp = call_tool(
                    tools["game.command"],  # type: ignore[index]
                    args=cmd,
                    manifest_path=args.tool_manifest,
                    timeout_sec=5.0,
                )
                print(f"[agent] say={say!r} action={action} -> game.command {cmd} resp={resp}")
        else:
            print(f"[agent] say={say!r} action={action} (no game.command tool configured)")

        time.sleep(max(0.05, float(args.tick_sec)))


if __name__ == "__main__":
    main()

