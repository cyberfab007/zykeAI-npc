"""
Simple policy service stub.
Loads the latest checkpoint from models/checkpoints and serves /get_action and /current_policy_version.
"""
import os
from functools import wraps
from typing import Dict, Optional

import torch
from flask import Flask, jsonify, request

from policy_service.weights_manager import WeightsManager
from trainer.models import MLPPolicy

API_TOKEN = os.getenv("API_TOKEN")  # optional bearer token

app = Flask(__name__)
weights = WeightsManager(checkpoint_dir=os.getenv("CHECKPOINT_DIR", "models/checkpoints"))


def make_model():
    # TODO: parameterize obs_dim/action_dim via config
    return MLPPolicy(obs_dim=128, action_dim=32)


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if API_TOKEN:
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {API_TOKEN}":
                return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)

    return wrapper


@app.route("/current_policy_version", methods=["GET"])
def current_policy_version():
    return jsonify({"policy_version": weights.policy_version})


@app.route("/get_action", methods=["POST"])
@require_auth
def get_action():
    payload = request.get_json(force=True, silent=True) or {}
    obs = payload.get("obs")
    if obs is None:
        return jsonify({"error": "missing obs"}), 400

    # Refresh weights if a newer checkpoint exists
    weights.load_latest(make_model)

    if weights.model is None:
        return jsonify({"error": "no model loaded"}), 503

    with torch.no_grad():
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        logits, value = weights.model(obs_tensor)
        action = int(torch.argmax(logits, dim=-1).item())

    return jsonify(
        {
            "action": action,
            "policy_version": weights.policy_version,
            "value_estimate": float(value.item()),
        }
    )


if __name__ == "__main__":
    weights.load_latest(make_model)
    app.run(host="0.0.0.0", port=8000)
