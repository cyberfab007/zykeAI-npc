"""
Worker client for the trainer service.

Modes:
- mlp: legacy demo mode (MLPPolicy) that trains on synthetic data.
- llm: LoRA-on-LLM mode that trains adapter deltas on "experience blocks"
       (see data/make_experience_blocks.py) and submits LoRA parameter deltas.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.adapter_manifest import select_adapter
from trainer.models import MLPPolicy, make_optimizer


def parse_args():
    p = argparse.ArgumentParser(description="Worker to exercise trainer endpoints with real deltas.")
    p.add_argument("--trainer-url", default=os.getenv("TRAINER_URL", "http://localhost:5001"))
    p.add_argument("--node-id", default=os.getenv("NODE_ID", "local"))
    p.add_argument("--mode", default=os.getenv("WORKER_MODE", "mlp"), choices=["mlp", "llm"])
    p.add_argument("--num-tasks", type=int, default=int(os.getenv("NUM_TASKS", "3")))
    p.add_argument("--sleep", type=float, default=float(os.getenv("SLEEP_SEC", "0.5")))
    # MLP mode dims
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8, help="Default batch_size for local steps.")
    # LLM mode config
    p.add_argument("--adapter-name", default=os.getenv("ADAPTER_NAME"))
    p.add_argument("--manifest-path", default=os.getenv("MANIFEST_PATH", "data/adapters/manifest.json"))
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL"))
    p.add_argument("--tokenizer", default=os.getenv("TOKENIZER"))
    return p.parse_args()


def snapshot_trainable(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: p.detach().to("cpu").clone() for k, p in model.named_parameters() if p.requires_grad}


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


def fetch_task(trainer_url: str, node_id: str) -> Dict:
    r = requests.get(
        f"{trainer_url}/get_task",
        params={"node_id": node_id},
        headers={"X-Node-Id": node_id},
        timeout=10,
    )
    r.raise_for_status()
    body = r.json()
    return body.get("task") if isinstance(body, dict) else body


def fetch_weights(trainer_url: str, version: int) -> Dict[str, torch.Tensor]:
    r = requests.get(f"{trainer_url}/get_lora_weights", params={"version": version}, timeout=20)
    r.raise_for_status()
    data = r.json()
    state_b64 = data["lora_state_b64"]
    buf = io.BytesIO(base64.b64decode(state_b64.encode("ascii")))
    return torch.load(buf, map_location="cpu")


def fetch_block(trainer_url: str, block_id: str) -> Dict:
    r = requests.get(f"{trainer_url}/get_block", params={"block_id": block_id}, timeout=20)
    r.raise_for_status()
    return r.json().get("block") or {}


def _apply_trainable_state(model: torch.nn.Module, trainable_state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in trainable_state:
                param.copy_(trainable_state[name].to(device=param.device, dtype=param.dtype))


_LLM_CACHE: Dict[Tuple, Tuple[torch.nn.Module, object, torch.device]] = {}


def get_llm_with_lora(base_model: str, tokenizer_name: str, adapter_entry: Dict):
    """
    Cached base model + LoRA wrapper. Per-task weights are applied via _apply_trainable_state().
    """
    key = (
        base_model,
        tokenizer_name,
        tuple(adapter_entry.get("target_modules") or []),
        int(adapter_entry.get("r") or 16),
        int(adapter_entry.get("lora_alpha") or 32),
        float(adapter_entry.get("lora_dropout") or 0.05),
    )
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for --mode llm") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device.type == "cuda" else None)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, torch_dtype=dtype)
    base.to(device)

    target_modules = adapter_entry.get("target_modules") or []
    if not target_modules:
        raise RuntimeError(f"Adapter entry missing target_modules for llm mode: {adapter_entry.get('name')}")

    lora_cfg = LoraConfig(
        r=int(adapter_entry.get("r") or 16),
        lora_alpha=int(adapter_entry.get("lora_alpha") or 32),
        lora_dropout=float(adapter_entry.get("lora_dropout") or 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    _LLM_CACHE[key] = (model, tokenizer, device)
    return model, tokenizer, device


def train_one_task_mlp(task: Dict, trainer_url: str, args) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(args.obs_dim, args.action_dim).to(device)
    lora_state = fetch_weights(trainer_url, int(task["model_version"]))
    state_dict = model.state_dict()
    state_dict.update(lora_state)
    model.load_state_dict(state_dict)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, lr=float(task.get("learning_rate", 1e-4)))

    old_state = snapshot_trainable(model)
    steps = int(task.get("num_steps", 50))
    batch_size = int(task.get("batch_size", args.batch_size))

    loss_sum = 0.0
    grad_norm_sum = 0.0
    steps_done = 0
    t0 = time.time()

    for _ in range(steps):
        obs = torch.randn(batch_size, args.obs_dim, device=device)
        targets = torch.randint(0, args.action_dim, (batch_size,), device=device)
        logits, _ = model(obs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += float(loss.item())
        grad_norm_sum += float(grad_norm)
        steps_done += 1

    new_state = snapshot_trainable(model)
    delta_state = compute_delta(new_state, old_state)
    delta_b64 = serialize_state(delta_state)

    metrics = {
        "train_loss_mean": loss_sum / max(1, steps_done),
        "train_loss_last": float(loss.item()),
        "grad_norm_mean": grad_norm_sum / max(1, steps_done),
        "steps_completed": steps_done,
        "duration_sec": time.time() - t0,
    }

    payload = {
        "task_id": task["task_id"],
        "base_model_version": int(task["model_version"]),
        "num_samples": steps_done * batch_size,
        "lora_delta": delta_b64,
        "metrics": metrics,
        "block_id": task.get("block_id"),
        "block_hash": task.get("block_hash"),
    }
    resp = requests.post(
        f"{trainer_url}/submit_update",
        json={**payload, "node_id": args.node_id},
        headers={"X-Node-Id": args.node_id},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _batch_from_steps(steps: List[Dict], tokenizer, device: torch.device, batch_size: int):
    import random

    if not steps:
        return None
    chosen = [steps[random.randrange(len(steps))] for _ in range(batch_size)]
    pad_id = int(tokenizer.pad_token_id)

    seqs = []
    labs = []
    for st in chosen:
        obs = st.get("obs") or []
        action = st.get("action")
        if action is None or not obs:
            continue
        ids = list(obs) + [int(action)]
        lab = ([-100] * len(obs)) + [int(action)]
        seqs.append(ids)
        labs.append(lab)
    if not seqs:
        return None

    max_len = max(len(s) for s in seqs)
    input_ids = []
    attention_mask = []
    labels = []
    for ids, lab in zip(seqs, labs):
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(lab + [-100] * pad_len)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def train_one_task_llm(task: Dict, trainer_url: str, args) -> Dict:
    adapter_name = task.get("target_adapter") or task.get("adapter_name") or args.adapter_name
    if not adapter_name:
        raise RuntimeError("LLM mode requires adapter_name (task.target_adapter or --adapter-name).")

    _adapter_path, entry = select_adapter(adapter_name, args.manifest_path)
    base_model = str(task.get("base_model") or args.base_model or entry.get("base_model"))
    tokenizer_name = str(task.get("tokenizer") or args.tokenizer or base_model)

    model, tokenizer, device = get_llm_with_lora(base_model, tokenizer_name, entry)

    # Sync current LoRA weights
    lora_state = fetch_weights(trainer_url, int(task["model_version"]))
    _apply_trainable_state(model, lora_state)

    block_id = task.get("block_id")
    if not block_id:
        raise RuntimeError("LLM mode requires block_id in task.")
    block = fetch_block(trainer_url, block_id)
    steps = block.get("steps") or []
    if not steps:
        raise RuntimeError(f"Block {block_id} has no steps.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(task.get("learning_rate", 1e-4)))

    old_state = snapshot_trainable(model)
    updates = int(task.get("num_steps", 50))
    batch_size = int(task.get("batch_size", 4))

    loss_sum = 0.0
    grad_norm_sum = 0.0
    steps_done = 0
    t0 = time.time()

    model.train()
    for _ in range(updates):
        batch = _batch_from_steps(steps, tokenizer, device, batch_size=batch_size)
        if batch is None:
            break
        input_ids, attention_mask, labels = batch
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
        optimizer.step()

        loss_sum += float(loss.item())
        grad_norm_sum += float(grad_norm)
        steps_done += 1

    model.eval()

    new_state = snapshot_trainable(model)
    delta_state = compute_delta(new_state, old_state)
    delta_b64 = serialize_state(delta_state)

    metrics = {
        "train_loss_mean": loss_sum / max(1, steps_done),
        "train_loss_last": float(loss.item()) if steps_done else None,
        "grad_norm_mean": grad_norm_sum / max(1, steps_done),
        "steps_completed": steps_done,
        "duration_sec": time.time() - t0,
        "adapter_name": adapter_name,
        "base_model": base_model,
    }

    payload = {
        "task_id": task["task_id"],
        "base_model_version": int(task["model_version"]),
        "num_samples": steps_done * batch_size,
        "lora_delta": delta_b64,
        "metrics": metrics,
        "block_id": block_id,
        "block_hash": task.get("block_hash"),
    }
    resp = requests.post(
        f"{trainer_url}/submit_update",
        json={**payload, "node_id": args.node_id},
        headers={"X-Node-Id": args.node_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def train_one_task(task: Dict, trainer_url: str, args) -> Dict:
    if args.mode == "llm" or (task.get("trainer_backend") == "llm"):
        return train_one_task_llm(task, trainer_url, args)
    return train_one_task_mlp(task, trainer_url, args)


def main():
    args = parse_args()
    stop_heartbeat = threading.Event()
    state_lock = threading.Lock()
    worker_state = {"state": "idle"}

    def heartbeat_loop():
        while not stop_heartbeat.is_set():
            try:
                with state_lock:
                    cur_state = worker_state["state"]
                requests.post(
                    f"{args.trainer_url}/node_heartbeat",
                    json={
                        "node_id": args.node_id,
                        "state": cur_state,
                        "capabilities": {
                            "mode": args.mode,
                            "device": "cuda" if torch.cuda.is_available() else "cpu",
                            "adapter_name": args.adapter_name,
                            "base_model": args.base_model,
                        },
                    },
                    timeout=5,
                )
            except Exception:
                pass
            stop_heartbeat.wait(2.0)
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    for _ in range(args.num_tasks):
        task = fetch_task(args.trainer_url, node_id=args.node_id)
        if not task:
            time.sleep(args.sleep)
            continue
        with state_lock:
            worker_state["state"] = "training"
        result = train_one_task(task, args.trainer_url, args)
        with state_lock:
            worker_state["state"] = "idle"
        print(
            f"Submitted task {task['task_id']} base_version={task['model_version']} -> "
            f"policy_version={result.get('current_policy_version')}"
        )
        time.sleep(args.sleep)
    stop_heartbeat.set()


if __name__ == "__main__":
    main()
