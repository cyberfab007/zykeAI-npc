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
    os.environ["REPLICAS_PER_BLOCK"] = "2"
    os.environ["BEST_K_PER_BLOCK"] = "1"

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

    # Heartbeat registers nodes.
    node_a = "node-a"
    node_b = "node-b"
    resp = client.post("/node_heartbeat", json={"node_id": node_a, "state": "idle", "capabilities": {"mode": "mlp"}})
    assert resp.status_code == 200
    resp = client.post("/node_heartbeat", json={"node_id": node_b, "state": "idle", "capabilities": {"mode": "mlp"}})
    assert resp.status_code == 200

    # Enqueue one block (will be replicated twice).
    blocks_path = _write_blocks_file(tmp_path, n=1)
    resp = client.post("/enqueue_blocks", json={"dataset_label": "test", "blocks_path": str(blocks_path)})
    assert resp.status_code == 200

    qs = client.get("/queue_status").get_json()
    assert qs["total"] == 1
    assert qs["pending"] == 1

    # Claim tasks for the same block (replicas=2): should include assignment_id, block_id and block_hash.
    task_a = client.get("/get_task", query_string={"node_id": node_a}, headers={"X-Node-Id": node_a}).get_json()["task"]
    task_b = client.get("/get_task", query_string={"node_id": node_b}, headers={"X-Node-Id": node_b}).get_json()["task"]
    assert task_a and task_a["block_id"].startswith("block-")
    assert task_b and task_b["block_id"] == task_a["block_id"]
    assert isinstance(task_a.get("assignment_id"), str) and task_a["assignment_id"]
    assert isinstance(task_b.get("assignment_id"), str) and task_b["assignment_id"] != task_a["assignment_id"]
    assert isinstance(task_a.get("block_hash"), str) and len(task_a["block_hash"]) == 64

    # Fetch the block bytes from trainer.
    blk = client.get("/get_block", query_string={"block_id": task_a["block_id"]}).get_json()["block"]
    assert blk["block_id"] == task_a["block_id"]
    assert blk["block_hash"] == task_a["block_hash"]

    # Build a tiny, non-zero delta in the correct shape.
    weights = _decode_weights(client.get("/get_lora_weights", query_string={"version": task_a["model_version"]}).get_json())
    assert weights
    first_key = next(iter(weights.keys()))
    delta1 = {k: torch.zeros_like(v, dtype=torch.float16) for k, v in weights.items()}
    delta1[first_key].view(-1)[0] = torch.tensor(1e-2, dtype=torch.float16)
    delta_b64_1 = _encode_delta(delta1)
    delta2 = {k: torch.zeros_like(v, dtype=torch.float16) for k, v in weights.items()}
    delta2[first_key].view(-1)[0] = torch.tensor(2e-2, dtype=torch.float16)
    delta_b64_2 = _encode_delta(delta2)

    payload_a = {
        "task_id": task_a["task_id"],
        "base_model_version": task_a["model_version"],
        "num_samples": 8,
        "lora_delta": delta_b64_1,
        "metrics": {"train_loss_mean": 1.0, "train_loss_last": 1.0, "grad_norm_mean": 0.1},
        "assignment_id": task_a["assignment_id"],
        "block_id": task_a["block_id"],
        "block_hash": task_a["block_hash"],
        "node_id": node_a,
    }
    payload_b = {
        "task_id": task_b["task_id"],
        "base_model_version": task_b["model_version"],
        "num_samples": 8,
        "lora_delta": delta_b64_2,
        "metrics": {"train_loss_mean": 1.0, "train_loss_last": 1.0, "grad_norm_mean": 0.1},
        "assignment_id": task_b["assignment_id"],
        "block_id": task_b["block_id"],
        "block_hash": task_b["block_hash"],
        "node_id": node_b,
    }

    resp = client.post("/submit_update", json=payload_a, headers={"X-Node-Id": node_a})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["accepted"] is True
    # Not decided until replica 2 arrives.
    assert body["current_policy_version"] == task_a["model_version"]

    resp = client.post("/submit_update", json=payload_b, headers={"X-Node-Id": node_b})
    assert resp.status_code == 200
    body2 = resp.get_json()
    assert body2["accepted"] is True
    assert body2["current_policy_version"] == task_a["model_version"] + 1  # block decision submitted 1 update; round size=1

    qs2 = client.get("/queue_status").get_json()
    assert qs2["completed"] == 1

    # Cluster should show the node and blocks_completed incremented.
    cluster = client.get("/cluster_status").get_json()
    nodes = {n["node_id"]: n for n in cluster["nodes"]}
    assert node_a in nodes
    assert node_b in nodes

    # Disable node prevents new assignments.
    resp = client.post("/disable_node", json={"node_id": node_a})
    assert resp.status_code == 200
    task2 = client.get("/get_task", query_string={"node_id": node_a}, headers={"X-Node-Id": node_a}).get_json()["task"]
    assert task2 is None

    # Events SSE should produce backlog lines.
    resp = client.get("/events", buffered=False)
    first_chunk = next(resp.response).decode("utf-8", errors="ignore")
    assert "data:" in first_chunk
    assert "trainer_initialized" in first_chunk or "blocks_enqueued" in first_chunk
    resp.close()
