"""
Trainer service with task assignment and versioned aggregation for updates.
Aggregates updates per base_model_version, advances policy_version, and saves checkpoints.
"""
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

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

app = Flask(__name__)


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

    def submit_update(self, base_version: int, num_samples: int, delta_state: Dict[str, torch.Tensor]) -> bool:
        # Reject overly stale
        if base_version < self.current_version - MAX_STALENESS:
            return False
        self._ensure_round_initialized(base_version)
        self.pending_updates[base_version].append(
            {"base_version": base_version, "num_samples": num_samples, "delta_state": delta_state}
        )
        self._maybe_aggregate_version(base_version)
        return True

    def tick(self):
        self._maybe_aggregate_version(self.current_version)

    def _maybe_aggregate_version(self, version: int) -> None:
        if version != self.current_version:
            return
        updates = self.pending_updates.get(version, [])
        n_updates = len(updates)
        if n_updates == 0:
            return
        if n_updates >= NUM_TASKS_PER_ROUND:
            self._aggregate_and_advance(version)
            return
        elapsed = time.time() - self.round_start_time.get(version, 0.0)
        if elapsed >= ROUND_TIMEOUT_SEC and n_updates >= MIN_UPDATES_PER_ROUND:
            self._aggregate_and_advance(version)

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
                    agg_delta[k] += weight * delta_tensor
        # Apply aggregated delta
        for k in self.global_state.keys():
            self.global_state[k] = self.global_state[k] + agg_delta[k]
        self.current_version = version + 1
        # Cleanup old
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


def init_trainer():
    global model, optimizer, aggregator
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    model = MLPPolicy(obs_dim=128, action_dim=32)
    optimizer = make_optimizer(model)
    loaded_version = load_latest_checkpoint(model, optimizer, CHECKPOINT_DIR)
    current_version = loaded_version if loaded_version is not None else 0
    aggregator = VersionedAggregator(model.state_dict(), current_version)
    if loaded_version is None:
        save_checkpoint(model, optimizer, current_version, CHECKPOINT_DIR)


@app.route("/get_task", methods=["GET"])
def get_task():
    with state_lock:
        task = {"task_id": str(uuid.uuid4()), "model_version": aggregator.current_version}
    return jsonify(task)


@app.route("/submit_update", methods=["POST"])
def submit_update():
    payload = request.get_json(force=True, silent=True) or {}
    required = ["task_id", "base_model_version", "delta_scale"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400
    base_version = int(payload["base_model_version"])
    delta_scale = float(payload["delta_scale"])
    # For stub: synthesize a delta_state by scaling current weights
    with state_lock:
        current_state = aggregator.get_current_state()
        delta_state = {k: delta_scale * torch.zeros_like(v) for k, v in current_state.items()}
        accepted = aggregator.submit_update(base_version, num_samples=1, delta_state=delta_state)
        if not accepted:
            return jsonify({"error": "stale base_model_version"}), 400
        # After potential aggregation, write checkpoint for the current version
        save_checkpoint_from_state(aggregator)
    return jsonify({"status": "ok", "current_policy_version": aggregator.current_version})


def save_checkpoint_from_state(agg: VersionedAggregator):
    global model, optimizer
    # Load agg state into model, then save checkpoint
    model.load_state_dict(agg.get_current_state())
    save_checkpoint(model, optimizer, agg.current_version, CHECKPOINT_DIR)


if __name__ == "__main__":
    init_trainer()
    app.run(host="0.0.0.0", port=5001)
