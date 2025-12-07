"""
Minimal worker that:
- fetches a task
- downloads current trainable weights
- runs a tiny local training loop
- computes a real parameter delta
- submits the delta + metrics back to the trainer.
"""
import argparse
import base64
import io
import sys
import time
from pathlib import Path
from typing import Dict

import requests
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainer.models import MLPPolicy, make_optimizer


def parse_args():
    p = argparse.ArgumentParser(description="Worker to exercise trainer endpoints with real deltas.")
    p.add_argument("--trainer-url", default="http://localhost:5001")
    p.add_argument("--num-tasks", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep between tasks (seconds).")
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def snapshot_trainable(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}


def compute_delta(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for k, new_t in new_state.items():
        delta[k] = (new_t - old_state[k]).to(torch.float16)
    return delta


def serialize_state(state: Dict[str, torch.Tensor]) -> str:
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fetch_task(trainer_url: str):
    r = requests.get(f"{trainer_url}/get_task", timeout=5)
    r.raise_for_status()
    body = r.json()
    return body.get("task") if isinstance(body, dict) else body


def fetch_weights(trainer_url: str, version: int) -> Dict[str, torch.Tensor]:
    r = requests.get(f"{trainer_url}/get_lora_weights", params={"version": version}, timeout=5)
    r.raise_for_status()
    data = r.json()
    state_b64 = data["lora_state_b64"]
    buf = io.BytesIO(base64.b64decode(state_b64.encode("ascii")))
    return torch.load(buf, map_location="cpu")


def train_one_task(task, trainer_url: str, args):
    model = MLPPolicy(args.obs_dim, args.action_dim)
    # Load current trainable weights
    lora_state = fetch_weights(trainer_url, task["model_version"])
    state_dict = model.state_dict()
    state_dict.update(lora_state)
    model.load_state_dict(state_dict)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, lr=task.get("learning_rate", 1e-4))

    old_state = snapshot_trainable(model)
    steps = int(task.get("num_steps", 50))
    batch_size = int(task.get("batch_size", args.batch_size))

    loss_sum = 0.0
    grad_norm_sum = 0.0
    steps_done = 0
    t0 = time.time()

    for _ in range(steps):
        obs = torch.randn(batch_size, args.obs_dim)
        targets = torch.randint(0, args.action_dim, (batch_size,))
        logits, _ = model(obs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += loss.item()
        grad_norm_sum += grad_norm
        steps_done += 1

    new_state = snapshot_trainable(model)
    delta_state = compute_delta(new_state, old_state)
    delta_b64 = serialize_state(delta_state)

    metrics = {
        "train_loss_mean": loss_sum / max(1, steps_done),
        "train_loss_last": loss.item(),
        "grad_norm_mean": grad_norm_sum / max(1, steps_done),
        "steps_completed": steps_done,
        "duration_sec": time.time() - t0,
    }

    num_samples = steps_done * batch_size
    payload = {
        "task_id": task["task_id"],
        "base_model_version": task["model_version"],
        "num_samples": num_samples,
        "lora_delta": delta_b64,
        "metrics": metrics,
    }
    resp = requests.post(f"{trainer_url}/submit_update", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    args = parse_args()
    for _ in range(args.num_tasks):
        task = fetch_task(args.trainer_url)
        if not task:
            time.sleep(args.sleep)
            continue
        result = train_one_task(task, args.trainer_url, args)
        print(
            f"Submitted task {task['task_id']} base_version={task['model_version']} -> "
            f"policy_version={result.get('current_policy_version')}"
        )
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
