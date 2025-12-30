import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path

import pytest


pytest.importorskip("torch")
torch = pytest.importorskip("torch")


def _make_block(block_id: str) -> dict:
    block = {
        "block_id": block_id,
        "policy_version": 0,
        "env_id": "offline_corpus",
        "npc_type": "generic",
        "steps": [
            {"obs": [1, 2, 3, 4], "action": 5, "reward": 0.0, "done": False},
            {"obs": [2, 3, 4, 5], "action": 6, "reward": 0.0, "done": True},
        ],
    }
    canonical = json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")
    block["block_hash"] = hashlib.sha256(canonical).hexdigest()
    return block


def _write_blocks_file(tmp_path: Path, n: int = 2) -> Path:
    p = tmp_path / "blocks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps(_make_block(f"block-{i}")) + "\n")
    return p


def _load_trainer(tmp_path: Path):
    os.environ["TRAINER_BACKEND"] = "mlp"
    os.environ["TRAINER_DB_PATH"] = str(tmp_path / "trainer.db")
    os.environ["CHECKPOINT_DIR"] = str(tmp_path / "checkpoints")
    os.environ["NUM_TASKS_PER_ROUND"] = "1"
    os.environ["MIN_UPDATES_PER_ROUND"] = "1"
    os.environ["ROUND_TIMEOUT_SEC"] = "999"
    os.environ["REQUIRE_BLOCK_HASH"] = "true"
    os.environ["TICK_INTERVAL_SEC"] = "999"

    import trainer.server as server

    importlib.reload(server)
    server.init_trainer()
    return server


def _decode_weights(resp_json: dict) -> dict:
    b64 = resp_json["lora_state_b64"]
    buf = io.BytesIO(base64.b64decode(b64.encode("ascii")))
    return torch.load(buf, map_location="cpu")


def _encode_delta(delta_state: dict) -> str:
    buf = io.BytesIO()
    torch.save(delta_state, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def test_c2_queue_assign_submit_persist_and_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = _load_trainer(tmp_path)
    client = server.app.test_client()

    # Heartbeat registers node.
    node_id = "node-a"
    resp = client.post("/node_heartbeat", json={"node_id": node_id, "state": "idle", "capabilities": {"mode": "mlp"}})
    assert resp.status_code == 200

    # Enqueue a couple blocks.
    blocks_path = _write_blocks_file(tmp_path, n=2)
    resp = client.post("/enqueue_blocks", json={"dataset_label": "test", "blocks_path": str(blocks_path)})
    assert resp.status_code == 200

    qs = client.get("/queue_status").get_json()
    assert qs["total"] == 2
    assert qs["pending"] == 2

    # Claim a task: should include block_id and block_hash.
    task = client.get("/get_task", query_string={"node_id": node_id}, headers={"X-Node-Id": node_id}).get_json()["task"]
    assert task and task["block_id"].startswith("block-")
    assert isinstance(task.get("block_hash"), str) and len(task["block_hash"]) == 64

    # Fetch the block bytes from trainer.
    blk = client.get("/get_block", query_string={"block_id": task["block_id"]}).get_json()["block"]
    assert blk["block_id"] == task["block_id"]
    assert blk["block_hash"] == task["block_hash"]

    # Build a tiny, non-zero delta in the correct shape.
    weights = _decode_weights(client.get("/get_lora_weights", query_string={"version": task["model_version"]}).get_json())
    assert weights
    first_key = next(iter(weights.keys()))
    delta = {k: torch.zeros_like(v, dtype=torch.float16) for k, v in weights.items()}
    delta[first_key].view(-1)[0] = torch.tensor(1e-2, dtype=torch.float16)
    delta_b64 = _encode_delta(delta)

    payload = {
        "task_id": task["task_id"],
        "base_model_version": task["model_version"],
        "num_samples": 8,
        "lora_delta": delta_b64,
        "metrics": {"train_loss_mean": 1.0, "train_loss_last": 1.0, "grad_norm_mean": 0.1},
        "block_id": task["block_id"],
        "block_hash": task["block_hash"],
        "node_id": node_id,
    }
    resp = client.post("/submit_update", json=payload, headers={"X-Node-Id": node_id})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["accepted"] is True
    assert body["current_policy_version"] == task["model_version"] + 1  # NUM_TASKS_PER_ROUND=1 triggers aggregation

    qs2 = client.get("/queue_status").get_json()
    assert qs2["completed"] == 1

    # Cluster should show the node and blocks_completed incremented.
    cluster = client.get("/cluster_status").get_json()
    nodes = {n["node_id"]: n for n in cluster["nodes"]}
    assert node_id in nodes
    assert int(nodes[node_id]["blocks_completed"]) >= 1

    # Disable node prevents new assignments.
    resp = client.post("/disable_node", json={"node_id": node_id})
    assert resp.status_code == 200
    task2 = client.get("/get_task", query_string={"node_id": node_id}, headers={"X-Node-Id": node_id}).get_json()["task"]
    assert task2 is None

    # Events SSE should produce backlog lines.
    resp = client.get("/events", buffered=False)
    first_chunk = next(resp.response).decode("utf-8", errors="ignore")
    assert "data:" in first_chunk
    assert "trainer_initialized" in first_chunk or "blocks_enqueued" in first_chunk
    resp.close()

