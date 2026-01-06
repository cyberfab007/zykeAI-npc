# NPC Runtime

This folder is for **agent loops** (the “always thinking” NPC) built on:
- LoRA adapters (persona/behavior modules)
- MCP-style tools (`data/mcp/tools.json`) for world state + actions

The key idea is a repeatable loop:
1) Sense (MCP tools)
2) Think (model + adapter)
3) Act (MCP tools)
4) Persist memory

## Quickstart (local stub)

1) Start the MCP tool server stub:

```bash
python -m src.mcp.server_stub
```

2) Run the dog_guard agent loop (dry-run by default; it prints intended commands):

```bash
python -m npc.agent_loop \
  --npc-type dog_guard \
  --adapter-name npc_core_pythia_410m_v1 \
  --base-model EleutherAI/pythia-410m-deduped \
  --tick-sec 1.0 \
  --dry-run
```

3) To execute actions, remove `--dry-run` (the stub MCP server will accept them).

## Notes

- For the metaverse/game, replace the stub MCP server handlers with real game state + command execution.
- Adapters are base-model specific; choose `adapter_name` that matches the `base_model`.

