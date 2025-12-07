"""
Minimal trainer service stub exposing /get_task and /submit_update.
It aggregates a fixed number of dummy updates per round, bumps policy_version,
saves a new checkpoint, and relies on policy_service to serve the latest weights.
"""
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List

import torch
from flask import Flask, jsonify, request

from trainer.checkpointing import save_checkpoint
from trainer.models import MLPPolicy, make_optimizer
from trainer.weights import load_latest_checkpoint

NUM_TASKS_PER_ROUND = int(os.getenv("NUM_TASKS_PER_ROUND", "3"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "models/checkpoints")

app = Flask(__name__)

# Global trainer state (for stub purposes)
state_lock = threading.Lock()
policy_version: int = 0
model: torch.nn.Module
optimizer: torch.optim.Optimizer
pending_updates: List[Dict] = []


def init_model():
    global model, optimizer, policy_version
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    model = MLPPolicy(obs_dim=128, action_dim=32)
    optimizer = make_optimizer(model)
    loaded_version = load_latest_checkpoint(model, optimizer, CHECKPOINT_DIR)
    if loaded_version is not None:
        policy_version = loaded_version
    else:
        policy_version = 0
        save_checkpoint(model, optimizer, policy_version, CHECKPOINT_DIR)


@app.route("/get_task", methods=["GET"])
def get_task():
    with state_lock:
        task = {
            "task_id": str(uuid.uuid4()),
            "model_version": policy_version,
        }
    return jsonify(task)


def aggregate_and_save():
    global policy_version, pending_updates
    if not pending_updates:
        return
    mean_delta = sum(u["delta_scale"] for u in pending_updates) / len(pending_updates)
    # Apply a small noise scaled by mean_delta to simulate an update
    with torch.no_grad():
        for p in model.parameters():
            p.add_(mean_delta * 0.001 * torch.randn_like(p))
    policy_version += 1
    save_checkpoint(model, optimizer, policy_version, CHECKPOINT_DIR)
    pending_updates = []
    app.logger.info(f"Aggregated {NUM_TASKS_PER_ROUND} updates -> new policy_version {policy_version}")


@app.route("/submit_update", methods=["POST"])
def submit_update():
    payload = request.get_json(force=True, silent=True) or {}
    required = ["task_id", "base_model_version", "delta_scale"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400
    base_version = int(payload["base_model_version"])
    delta_scale = float(payload["delta_scale"])
    with state_lock:
        if base_version != policy_version:
            return jsonify({"error": "stale base_model_version"}), 400
        pending_updates.append({"task_id": payload["task_id"], "delta_scale": delta_scale})
        if len(pending_updates) >= NUM_TASKS_PER_ROUND:
            aggregate_and_save()
    return jsonify({"status": "ok", "current_policy_version": policy_version})


if __name__ == "__main__":
    init_model()
    app.run(host="0.0.0.0", port=5001)
