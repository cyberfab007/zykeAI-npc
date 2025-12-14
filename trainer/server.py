"""
Trainer service with task assignment and versioned aggregation for updates.
Aggregates LoRA-like deltas per base_model_version, advances policy_version, and saves checkpoints.
"""
import base64
import io
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from flask import Flask, jsonify, request

from trainer.checkpointing import save_checkpoint
from trainer.models import MLPPolicy, make_optimizer
from trainer.update_validation import UpdateRejected, trimmed_mean, validate_update
from trainer.weights import load_latest_checkpoint

NUM_TASKS_PER_ROUND = int(os.getenv("NUM_TASKS_PER_ROUND", "3"))
MIN_UPDATES_PER_ROUND = int(os.getenv("MIN_UPDATES_PER_ROUND", "1"))
ROUND_TIMEOUT_SEC = float(os.getenv("ROUND_TIMEOUT_SEC", "30"))
MAX_STALENESS = int(os.getenv("MAX_STALENESS", "1"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "models/checkpoints")
DELTA_NORM_MAX = float(os.getenv("DELTA_NORM_MAX", "1e9"))
TICK_INTERVAL_SEC = float(os.getenv("TICK_INTERVAL_SEC", "1.0"))
MIN_DELTA_NORM = float(os.getenv("MIN_DELTA_NORM", "1e-8"))
TRIM_FRAC = float(os.getenv("TRIM_FRAC", "0.1"))
REQUIRE_BLOCK_HASH = os.getenv("REQUIRE_BLOCK_HASH", "false").lower() == "true"
BLOCK_HASH_MAP = os.getenv("BLOCK_HASH_MAP")  # path to block_id->hash mapping (JSON)

app = Flask(__name__)


def get_trainable_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}


def decode_delta(delta_b64: str) -> Dict[str, torch.Tensor]:
    delta_bytes = base64.b64decode(delta_b64.encode("ascii"))
    buffer = io.BytesIO(delta_bytes)
    delta_state = torch.load(buffer, map_location="cpu")
    return delta_state


def validate_delta(delta_state: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor]) -> bool:
    for k, v in delta_state.items():
        if k not in global_state:
            return False
        if v.shape != global_state[k].shape:
            return False
    return True


def delta_l2_norm(delta_state: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for t in delta_state.values():
        total += (t.float() ** 2).sum().item()
    return math.sqrt(total)


class VersionedAggregator:
    def __init__(self, initial_global_state: Dict[str, torch.Tensor], initial_version: int):
        self.global_state = {k: v.clone() for k, v in initial_global_state.items()}
        self.current_version = initial_version
        self.pending_updates: Dict[int, List[Dict]] = {}
        self.round_start_time: Dict[int, float] = {}
        self._ensure_round_initialized(self.current_version)

    def _ensure_round_initialized(self, version: int) -> None:
        if version not in self.pending_updates:
            self.pending_updates[version] = []
            self.round_start_time[version] = time.time()

    def submit_update(
        self, base_version: int, num_samples: int, delta_state: Dict[str, torch.Tensor], metrics: Dict
    ) -> Tuple[bool, bool]:
        """
        Returns (accepted, aggregated_now).
        """
        if base_version < self.current_version - MAX_STALENESS:
            return False, False
        self._ensure_round_initialized(base_version)
        self.pending_updates[base_version].append(
            {
                "base_version": base_version,
                "num_samples": num_samples,
                "delta_state": delta_state,
                "metrics": metrics,
            }
        )
        aggregated = self._maybe_aggregate_version(base_version)
        return True, aggregated

    def tick(self) -> bool:
        return self._maybe_aggregate_version(self.current_version)

    def _maybe_aggregate_version(self, version: int) -> bool:
        if version != self.current_version:
            return False
        updates = self.pending_updates.get(version, [])
        n_updates = len(updates)
        if n_updates == 0:
            return False
        if n_updates >= NUM_TASKS_PER_ROUND:
            self._aggregate_and_advance(version)
            return True
        elapsed = time.time() - self.round_start_time.get(version, 0.0)
        if elapsed >= ROUND_TIMEOUT_SEC and n_updates >= MIN_UPDATES_PER_ROUND:
            self._aggregate_and_advance(version)
            return True
        return False

    def _aggregate_and_advance(self, version: int) -> None:
        updates = self.pending_updates.get(version, [])
        if not updates:
            return
        # Robust aggregation: trimmed mean with norm clipping
        clipped_updates = []
        for u in updates:
            clipped = {}
            for k, t in u["delta_state"].items():
                clipped[k] = torch.clamp(t, min=-1e3, max=1e3)
            clipped_updates.append(clipped)
        agg_delta = trimmed_mean(clipped_updates, trim_frac=TRIM_FRAC)
        for k in self.global_state.keys():
            if k in agg_delta:
                self.global_state[k] = self.global_state[k] + agg_delta[k].to(self.global_state[k].dtype)
        self.current_version = version + 1
        del self.pending_updates[version]
        del self.round_start_time[version]
        self._ensure_round_initialized(self.current_version)

    def get_current_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.global_state.items()}


# Global trainer state
state_lock = threading.Lock()
model: torch.nn.Module
optimizer: torch.optim.Optimizer
aggregator: VersionedAggregator
_ticker_thread: threading.Thread
block_hashes: Dict[str, str] = {}
task_history: List[Dict] = []
queue_records: List[Dict] = []
cluster_nodes: Dict[str, Dict] = {}


def init_trainer():
    global model, optimizer, aggregator, _ticker_thread, block_hashes, task_history, queue_records, cluster_nodes
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    model = MLPPolicy(obs_dim=128, action_dim=32)
    optimizer = make_optimizer(model)
    loaded_version = load_latest_checkpoint(model, optimizer, CHECKPOINT_DIR)
    current_version = loaded_version if loaded_version is not None else 0
    aggregator = VersionedAggregator(get_trainable_state(model), current_version)
    if loaded_version is None:
        save_checkpoint(model, optimizer, current_version, CHECKPOINT_DIR)
    # Load block hash map if provided
    if BLOCK_HASH_MAP:
        try:
            import json

            with open(BLOCK_HASH_MAP, "r", encoding="utf-8") as f:
                block_hashes = json.load(f)
        except Exception:
            block_hashes = {}
    task_history = []
    queue_records = []
    cluster_nodes = {
        os.getenv("NODE_ID", "local"): {
            "node_id": os.getenv("NODE_ID", "local"),
            "state": "idle",
            "blocks_completed": 0,
            "trust_score": 1.0,
            "last_seen": time.strftime("%Y-%m-%d %H:%M:%S"),
            "enabled": True,
        }
    }
    _ticker_thread = threading.Thread(target=_tick_loop, daemon=True)
    _ticker_thread.start()


def apply_agg_state_to_model():
    global model
    trainable_state = aggregator.get_current_state()
    state_dict = model.state_dict()
    state_dict.update(trainable_state)
    model.load_state_dict(state_dict)


def _tick_loop():
    """Periodically trigger timeout-based aggregation."""
    while True:
        time.sleep(TICK_INTERVAL_SEC)
        with state_lock:
            aggregated = aggregator.tick()
            if aggregated:
                apply_agg_state_to_model()
                save_checkpoint(model, optimizer, aggregator.current_version, CHECKPOINT_DIR)


@app.route("/get_task", methods=["GET"])
def get_task():
    selected_block = None
    with state_lock:
        # Assign first pending block if available
        for rec in queue_records:
            if rec.get("status") == "pending":
                selected_block = rec
                rec["status"] = "assigned"
                rec["assigned_to"] = os.getenv("NODE_ID", "local")
                rec["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                break
    with state_lock:
        task = {
            "task_id": str(uuid.uuid4()),
            "model_version": aggregator.current_version,
            "data_shard_id": selected_block["block_id"] if selected_block else "stub",
            "num_steps": 100,
            "learning_rate": 1e-4,
            "batch_size": 1,
        }
        if selected_block:
            task["block_id"] = selected_block["block_id"]
            task["target_adapter"] = selected_block.get("target_adapter")
    return jsonify({"task": task})


@app.route("/get_lora_weights", methods=["GET"])
def get_lora_weights():
    version = request.args.get("version", default=None, type=int)
    with state_lock:
        if version is not None and version != aggregator.current_version:
            return jsonify({"error": "requested version not available"}), 400
        state = aggregator.get_current_state()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("ascii")
    return jsonify({"version": aggregator.current_version, "lora_state_b64": b64})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/worker_status", methods=["GET"])
def worker_status():
    with state_lock:
        state = "training" if aggregator.pending_updates.get(aggregator.current_version) else "idle"
        if not cluster_nodes.get(os.getenv("NODE_ID", "local"), {}).get("enabled", True):
            state = "disabled"
        blocks = []
        for entry in task_history[-10:]:
            blocks.append(
                {
                    "block_id": entry.get("block_id", ""),
                    "status": entry.get("status", ""),
                    "loss": entry.get("loss", ""),
                    "updated_at": entry.get("updated_at", ""),
                }
            )
        status = {
            "node_id": os.getenv("NODE_ID", "local"),
            "state": state,
            "base_model_version": aggregator.current_version if aggregator else None,
            "adapter_name": None,
            "quantization": None,
            "use_flash_attn": False,
            "compile_model": False,
            "blocks_processed": len(task_history),
            "blocks": blocks,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    return jsonify(status), 200


@app.route("/run_starter_test", methods=["POST"])
def run_starter_test():
    """
    Lightweight starter-block train/eval:
    - Reads data/starter_blocks/*.jsonl
    - Runs a tiny dummy MLP step on token counts (fast smoke)
    - Reports avg loss/time
    """
    root = Path("data/starter_blocks")
    if not root.exists():
        return jsonify({"ok": False, "message": "starter_blocks path not found"}), 400

    files = list(root.glob("*.jsonl"))
    if not files:
        return jsonify({"ok": False, "message": "no starter_block files found"}), 400

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(obs_dim=32, action_dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    max_steps = 50
    step = 0
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                if step >= max_steps:
                    break
                try:
                    import json

                    obj = json.loads(line)
                    text = (obj.get("input", "") + " " + obj.get("target", "")).strip()
                except Exception:
                    text = ""
                # simple feature: length mod some number
                feat = len(text) % 100
                obs = torch.tensor([[feat] + [0] * 31], dtype=torch.float32, device=device)
                logits, _ = model(obs)
                target = torch.tensor([feat % 8], dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                step += 1
                if step >= max_steps:
                    break

    elapsed = time.time() - start
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    return jsonify({"ok": True, "message": f"starter_test steps={len(losses)}, avg_loss={avg_loss:.4f}, time={elapsed:.2f}s"}), 200


@app.route("/enqueue_blocks", methods=["POST"])
def enqueue_blocks():
    payload = request.get_json(force=True, silent=True) or {}
    blocks_path = payload.get("blocks_path")
    dataset_label = payload.get("dataset_label", "unknown")
    target_adapter = payload.get("target_adapter")
    if not blocks_path or not Path(blocks_path).exists():
        return jsonify({"error": "blocks_path not found"}), 400
    added = 0
    with Path(blocks_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                obj = {}
            block_id = obj.get("block_id") or str(uuid.uuid4())
            queue_records.append(
                {
                    "block_id": block_id,
                    "task": dataset_label,
                    "target_adapter": target_adapter,
                    "status": "pending",
                    "hash": obj.get("block_hash"),
                    "assigned_to": None,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            added += 1
    return jsonify({"message": f"Enqueued {added} blocks from {blocks_path}"}), 200


@app.route("/queue_status", methods=["GET"])
def queue_status():
    with state_lock:
        pending = sum(1 for b in queue_records if b.get("status") == "pending")
        assigned = sum(1 for b in queue_records if b.get("status") == "assigned")
        completed = sum(1 for b in queue_records if b.get("status") == "completed")
        failed = sum(1 for b in queue_records if b.get("status") == "failed")
        total = len(queue_records)
        blocks = queue_records[-50:]
    return jsonify(
        {
            "total": total,
            "pending": pending,
            "assigned": assigned,
            "completed": completed,
            "failed": failed,
            "blocks": blocks,
        }
    )


@app.route("/cluster_status", methods=["GET"])
def cluster_status():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with state_lock:
        nodes = []
        for node_id, node in cluster_nodes.items():
            node["last_seen"] = now
            nodes.append(node.copy())
    return jsonify({"nodes": nodes})


@app.route("/disable_node", methods=["POST"])
def disable_node():
    payload = request.get_json(force=True, silent=True) or {}
    node_id = payload.get("node_id")
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    with state_lock:
        if node_id not in cluster_nodes:
            return jsonify({"error": "node not found"}), 404
        cluster_nodes[node_id]["enabled"] = False
        cluster_nodes[node_id]["state"] = "disabled"
    return jsonify({"message": f"Node {node_id} disabled"}), 200


@app.route("/enable_node", methods=["POST"])
def enable_node():
    payload = request.get_json(force=True, silent=True) or {}
    node_id = payload.get("node_id")
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    with state_lock:
        if node_id not in cluster_nodes:
            return jsonify({"error": "node not found"}), 404
        cluster_nodes[node_id]["enabled"] = True
        cluster_nodes[node_id]["state"] = "idle"
    return jsonify({"message": f"Node {node_id} enabled"}), 200


def metrics_valid(metrics: Dict) -> bool:
    if not metrics:
        return True
    for key in ["train_loss_mean", "train_loss_last", "grad_norm_mean"]:
        if key in metrics:
            val = metrics[key]
            if val is None or math.isnan(val) or math.isinf(val):
                return False
            if key.startswith("train_loss") and val > 1e4:
                return False
    return True


@app.route("/submit_update", methods=["POST"])
def submit_update():
    payload = request.get_json(force=True, silent=True) or {}
    required = ["task_id", "base_model_version", "num_samples", "lora_delta"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"accepted": False, "reason": f"missing fields: {', '.join(missing)}"}), 400

    base_version = int(payload["base_model_version"])
    num_samples = int(payload["num_samples"])
    metrics = payload.get("metrics", {})
    if not metrics_valid(metrics):
        return jsonify({"accepted": False, "reason": "metrics_invalid"}), 400

    block_id = payload.get("block_id")
    if REQUIRE_BLOCK_HASH:
        if "block_id" not in payload or "block_hash" not in payload:
            return jsonify({"accepted": False, "reason": "block_hash_required"}), 400
        block_id = payload["block_id"]
        submitted_hash = payload["block_hash"]
        canonical_hash = block_hashes.get(block_id)
        if canonical_hash is None:
            return jsonify({"accepted": False, "reason": "unknown_block_id"}), 400
    else:
        submitted_hash = None
        canonical_hash = None

    delta_state = decode_delta(payload["lora_delta"])
    try:
        validate_update(
            delta_state=delta_state,
            metrics=metrics,
            block_hash_submitted=payload.get("block_hash"),
            block_hash_canonical=canonical_hash,
            min_norm=MIN_DELTA_NORM,
            max_norm=DELTA_NORM_MAX,
            ref_grad=None,  # optional reference gradient
        )
    except UpdateRejected as exc:
        return jsonify({"accepted": False, "reason": str(exc)}), 400

    with state_lock:
        global_state = aggregator.get_current_state()
        if not validate_delta(delta_state, global_state):
            return jsonify({"accepted": False, "reason": "bad_delta_shape"}), 400
        accepted, aggregated = aggregator.submit_update(base_version, num_samples, delta_state, metrics)
        if not accepted:
            return jsonify({"accepted": False, "reason": "stale_version"}), 400
        # Record task in history
        task_history.append(
            {
                "block_id": block_id or "",
                "status": "submitted",
                "loss": metrics.get("train_loss_mean", None) if isinstance(metrics, dict) else None,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        if block_id:
            for rec in queue_records:
                if rec.get("block_id") == block_id:
                    rec["status"] = "completed"
                    rec["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    break
        # Save checkpoint if we aggregated this round
        if aggregated:
            apply_agg_state_to_model()
            save_checkpoint(model, optimizer, aggregator.current_version, CHECKPOINT_DIR)
    return jsonify({"accepted": True, "current_policy_version": aggregator.current_version})


if __name__ == "__main__":
    init_trainer()
    app.run(host="0.0.0.0", port=5001)
