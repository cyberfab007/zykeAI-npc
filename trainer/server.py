"""
Trainer service with task assignment and versioned aggregation for updates.
Aggregates LoRA-like deltas per base_model_version, advances policy_version, and saves checkpoints.
"""
import base64
import io
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
from trainer.weights import load_latest_checkpoint

NUM_TASKS_PER_ROUND = int(os.getenv("NUM_TASKS_PER_ROUND", "3"))
MIN_UPDATES_PER_ROUND = int(os.getenv("MIN_UPDATES_PER_ROUND", "1"))
ROUND_TIMEOUT_SEC = float(os.getenv("ROUND_TIMEOUT_SEC", "30"))
MAX_STALENESS = int(os.getenv("MAX_STALENESS", "1"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "models/checkpoints")
DELTA_NORM_MAX = float(os.getenv("DELTA_NORM_MAX", "1e9"))
TICK_INTERVAL_SEC = float(os.getenv("TICK_INTERVAL_SEC", "1.0"))

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
        agg_delta = {k: torch.zeros_like(v) for k, v in self.global_state.items()}
        total_samples = sum(u["num_samples"] for u in updates)
        if total_samples == 0:
            self.pending_updates[version] = []
            self.round_start_time[version] = time.time()
            return
        for u in updates:
            weight = float(u["num_samples"]) / float(total_samples)
            for k, delta_tensor in u["delta_state"].items():
                if k in agg_delta:
                    agg_delta[k] += weight * delta_tensor.to(agg_delta[k].dtype)
        for k in self.global_state.keys():
            self.global_state[k] = self.global_state[k] + agg_delta[k]
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


def init_trainer():
    global model, optimizer, aggregator, _ticker_thread
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    model = MLPPolicy(obs_dim=128, action_dim=32)
    optimizer = make_optimizer(model)
    loaded_version = load_latest_checkpoint(model, optimizer, CHECKPOINT_DIR)
    current_version = loaded_version if loaded_version is not None else 0
    aggregator = VersionedAggregator(get_trainable_state(model), current_version)
    if loaded_version is None:
        save_checkpoint(model, optimizer, current_version, CHECKPOINT_DIR)
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
    with state_lock:
        task = {
            "task_id": str(uuid.uuid4()),
            "model_version": aggregator.current_version,
            "data_shard_id": "stub",
            "num_steps": 100,
            "learning_rate": 1e-4,
            "batch_size": 1,
        }
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

    delta_state = decode_delta(payload["lora_delta"])
    delta_norm = delta_l2_norm(delta_state)
    if math.isnan(delta_norm) or math.isinf(delta_norm) or delta_norm > DELTA_NORM_MAX:
        return jsonify({"accepted": False, "reason": "delta_norm_invalid"}), 400

    with state_lock:
        global_state = aggregator.get_current_state()
        if not validate_delta(delta_state, global_state):
            return jsonify({"accepted": False, "reason": "bad_delta_shape"}), 400
        accepted, aggregated = aggregator.submit_update(base_version, num_samples, delta_state, metrics)
        if not accepted:
            return jsonify({"accepted": False, "reason": "stale_version"}), 400
        # Save checkpoint if we aggregated this round
        if aggregated:
            apply_agg_state_to_model()
            save_checkpoint(model, optimizer, aggregator.current_version, CHECKPOINT_DIR)
    return jsonify({"accepted": True, "current_policy_version": aggregator.current_version})


if __name__ == "__main__":
    init_trainer()
    app.run(host="0.0.0.0", port=5001)
