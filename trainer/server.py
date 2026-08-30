"""
Trainer service with task assignment and versioned aggregation for updates.
Aggregates LoRA-like deltas per base_model_version, advances policy_version, and saves checkpoints.
"""
import base64
import io
import json
import math
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from flask import Flask, Response, jsonify, request, stream_with_context

from trainer.checkpointing import save_checkpoint
from trainer.lora_checkpointing import load_latest_lora_checkpoint, save_lora_checkpoint
from trainer.db import (
    append_history,
    claim_next_block,
    enqueue_blocks as db_enqueue_blocks,
    get_assignment,
    get_block as db_get_block,
    get_block_meta,
    get_node,
    increment_node_blocks_completed,
    init_db,
    list_nodes,
    list_block_submissions,
    list_recent_blocks,
    mark_block_status,
    mark_assignment_status,
    queue_counts,
    recent_history,
    record_submission,
    set_block_decided,
    set_selected_submissions,
    set_node_enabled,
    upsert_node,
)
from trainer.models import MLPPolicy, make_optimizer
from trainer.update_validation import UpdateRejected, trimmed_mean, validate_update
from trainer.weights import load_latest_checkpoint

TRAINER_BACKEND = os.getenv("TRAINER_BACKEND", "mlp").lower()  # "mlp" | "llm"
TRAINER_DB_PATH = os.getenv("TRAINER_DB_PATH", "models/trainer.db")

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
REPLICAS_PER_BLOCK = int(os.getenv("REPLICAS_PER_BLOCK", "5"))
BEST_K_PER_BLOCK = int(os.getenv("BEST_K_PER_BLOCK", "1"))

app = Flask(__name__)

_event_clients: List[queue.Queue] = []
_event_clients_lock = threading.Lock()
_event_buffer: List[Dict] = []
_EVENT_BUFFER_MAX = int(os.getenv("EVENT_BUFFER_MAX", "200"))


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def emit_event(level: str, message: str, **fields):
    evt = {"ts": _now(), "level": level, "message": message, **fields}
    with _event_clients_lock:
        _event_buffer.append(evt)
        if len(_event_buffer) > _EVENT_BUFFER_MAX:
            del _event_buffer[: len(_event_buffer) - _EVENT_BUFFER_MAX]
        for q in list(_event_clients):
            try:
                q.put_nowait(evt)
            except Exception:
                # drop if client is too slow
                pass
    try:
        app.logger.info(evt)
    except Exception:
        pass


def get_trainable_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # Keep aggregation state on CPU to reduce GPU memory pressure and make serialization cheap.
    return {k: v.detach().to("cpu").clone() for k, v in model.named_parameters() if v.requires_grad}


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


def _score_mlp_delta(delta_state: Dict[str, torch.Tensor]) -> float:
    """
    Cheap server-side scoring for MLP backend:
    score = loss_before - loss_after on a fixed synthetic batch.
    """
    device = torch.device("cpu")
    obs_dim = 128
    action_dim = 32

    def eval_with_state(trainable_state: Dict[str, torch.Tensor]) -> float:
        tmp = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim).to(device)
        sd = tmp.state_dict()
        sd.update({k: v.to(device=device, dtype=sd[k].dtype) for k, v in trainable_state.items() if k in sd})
        tmp.load_state_dict(sd)
        tmp.eval()
        torch.manual_seed(0)
        obs = torch.randn(64, obs_dim, device=device)
        targets = torch.randint(0, action_dim, (64,), device=device)
        with torch.no_grad():
            logits, _ = tmp(obs)
            return float(torch.nn.functional.cross_entropy(logits, targets).item())

    base_state = aggregator.get_current_state()
    loss_before = eval_with_state(base_state)
    patched = {
        k: (base_state[k] + delta_state.get(k, torch.zeros_like(base_state[k]))).to(torch.float32)
        for k in base_state.keys()
    }
    loss_after = eval_with_state(patched)
    return loss_before - loss_after


def _cosine_similarity_delta(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, ta in a.items():
        tb = b.get(k)
        if tb is None:
            continue
        va = ta.float().view(-1)
        vb = tb.float().view(-1)
        dot += float((va * vb).sum().item())
        na += float((va * va).sum().item())
        nb += float((vb * vb).sum().item())
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _select_best_deltas(submissions: List[Dict[str, object]], best_k: int) -> Tuple[List[str], Dict[str, torch.Tensor], int]:
    """
    Decide winners among submissions for a block. Returns:
      (selected_submission_ids, aggregated_delta_state, total_samples)
    """
    decoded: List[Tuple[str, int, Dict[str, torch.Tensor], Dict]] = []
    for s in submissions:
        sid = str(s["submission_id"])
        num_samples = int(s.get("num_samples") or 0)
        delta_state = decode_delta(str(s["delta_b64"]))
        metrics = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
        decoded.append((sid, num_samples, delta_state, metrics))
    if not decoded:
        return [], {}, 0

    # Score each submission.
    scored: List[Tuple[float, str, int, Dict[str, torch.Tensor]]] = []
    if TRAINER_BACKEND == "mlp":
        for sid, ns, ds, _m in decoded:
            score = _score_mlp_delta(ds)
            scored.append((score, sid, ns, ds))
    else:
        # For LLM deltas, use consensus similarity to the mean delta.
        mean: Dict[str, torch.Tensor] = {}
        for _sid, _ns, ds, _m in decoded:
            for k, t in ds.items():
                mean[k] = mean.get(k, torch.zeros_like(t, dtype=torch.float32)) + t.float()
        for k in list(mean.keys()):
            mean[k] = mean[k] / float(len(decoded))
        for sid, ns, ds, _m in decoded:
            score = _cosine_similarity_delta(ds, mean)
            scored.append((score, sid, ns, ds))

    scored.sort(key=lambda x: x[0], reverse=True)
    winners = scored[: max(1, int(best_k))]
    total_samples = sum(w[2] for w in winners) or len(winners)
    agg: Dict[str, torch.Tensor] = {}
    for _score, _sid, ns, ds in winners:
        weight = float(ns) / float(total_samples) if total_samples else 1.0 / float(len(winners))
        for k, t in ds.items():
            agg[k] = agg.get(k, torch.zeros_like(t, dtype=torch.float32)) + (t.float() * weight)
    # store as fp16 to match other deltas
    agg_fp16 = {k: v.to(torch.float16) for k, v in agg.items()}
    selected_ids = [w[1] for w in winners]
    return selected_ids, agg_fp16, int(total_samples)


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
optimizer: torch.optim.Optimizer | None
aggregator: VersionedAggregator
_ticker_thread: threading.Thread
block_hashes: Dict[str, str] = {}
llm_base_model: str | None = None
llm_tokenizer_name: str | None = None
llm_adapter_name: str | None = None


def init_trainer():
    global model, optimizer, aggregator, _ticker_thread, block_hashes
    global llm_base_model, llm_tokenizer_name, llm_adapter_name
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    init_db(TRAINER_DB_PATH)
    upsert_node(
        TRAINER_DB_PATH,
        node_id=os.getenv("NODE_ID", "trainer"),
        state="idle",
        last_seen=_now(),
        enabled=True,
        capabilities={"role": "trainer", "backend": TRAINER_BACKEND, "device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    if TRAINER_BACKEND == "llm":
        # LLM LoRA delta backend: maintain trainable LoRA weights only (no full-base checkpointing).
        llm_adapter_name = os.getenv("LLM_ADAPTER_NAME") or os.getenv("ADAPTER_NAME")
        manifest_path = os.getenv("LLM_MANIFEST_PATH", "data/adapters/manifest.json")
        if not llm_adapter_name:
            # Default: open, small baseline for iteration.
            llm_adapter_name = "npc_core_pythia_410m_v1"

        try:
            from src.models.adapter_manifest import select_adapter

            _adapter_path, entry = select_adapter(llm_adapter_name, manifest_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load adapter manifest entry '{llm_adapter_name}': {exc}") from exc

        llm_base_model = str(entry.get("base_model") or os.getenv("LLM_BASE_MODEL") or "EleutherAI/pythia-410m-deduped")
        llm_tokenizer_name = os.getenv("LLM_TOKENIZER") or llm_base_model
        target_modules = entry.get("target_modules") or []
        r = int(entry.get("r") or os.getenv("LLM_LORA_R", "16"))
        alpha = int(entry.get("lora_alpha") or os.getenv("LLM_LORA_ALPHA", "32"))
        dropout = float(entry.get("lora_dropout") or os.getenv("LLM_LORA_DROPOUT", "0.05"))

        from transformers import AutoModelForCausalLM

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required for TRAINER_BACKEND=llm") from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device.type == "cuda" else None)
        base = AutoModelForCausalLM.from_pretrained(llm_base_model, trust_remote_code=True, torch_dtype=dtype)
        base.to(device)
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        optimizer = None

        loaded = load_latest_lora_checkpoint(CHECKPOINT_DIR)
        if loaded:
            loaded_version, lora_state, _meta = loaded
            # Apply loaded trainable weights onto the model.
            state_dict = model.state_dict()
            state_dict.update(lora_state)
            model.load_state_dict(state_dict)
            current_version = loaded_version
        else:
            current_version = 0
        aggregator = VersionedAggregator(get_trainable_state(model), current_version)
        if not loaded:
            save_lora_checkpoint(
                lora_state=aggregator.get_current_state(),
                version=current_version,
                output_dir=CHECKPOINT_DIR,
                metadata={"adapter_name": llm_adapter_name, "base_model": llm_base_model},
            )
    else:
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
    _ticker_thread = threading.Thread(target=_tick_loop, daemon=True)
    _ticker_thread.start()
    emit_event("info", "trainer_initialized", backend=TRAINER_BACKEND, checkpoint_dir=CHECKPOINT_DIR, db=TRAINER_DB_PATH)


def apply_agg_state_to_model():
    global model
    trainable_state = aggregator.get_current_state()
    # Avoid materializing a full state_dict for large base LMs; only patch trainable params.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in trainable_state:
                src = trainable_state[name].to(device=param.device, dtype=param.dtype)
                param.copy_(src)


def _tick_loop():
    """Periodically trigger timeout-based aggregation."""
    while True:
        time.sleep(TICK_INTERVAL_SEC)
        with state_lock:
            aggregated = aggregator.tick()
            if aggregated:
                apply_agg_state_to_model()
                if TRAINER_BACKEND == "llm":
                    save_lora_checkpoint(
                        lora_state=aggregator.get_current_state(),
                        version=aggregator.current_version,
                        output_dir=CHECKPOINT_DIR,
                        metadata={"adapter_name": llm_adapter_name, "base_model": llm_base_model},
                    )
                else:
                    assert optimizer is not None
                    save_checkpoint(model, optimizer, aggregator.current_version, CHECKPOINT_DIR)
                emit_event("info", "aggregated", model_version=aggregator.current_version)


@app.route("/get_task", methods=["GET"])
def get_task():
    node_id = request.headers.get("X-Node-Id") or request.args.get("node_id") or "unknown"
    node = get_node(TRAINER_DB_PATH, node_id)
    if node and not bool(node.get("enabled")):
        return jsonify({"task": None}), 200

    selected = claim_next_block(TRAINER_DB_PATH, node_id=node_id, updated_at=_now())
    with state_lock:
        task = {
            "task_id": str(uuid.uuid4()),
            "model_version": aggregator.current_version,
            "data_shard_id": selected["block_id"] if selected else "stub",
            "num_steps": 100,
            "learning_rate": 1e-4,
            "batch_size": 1,
        }
        if selected:
            task["assignment_id"] = selected.get("assignment_id")
            task["block_id"] = selected["block_id"]
            task["block_hash"] = selected.get("hash")
            task["target_adapter"] = selected.get("target_adapter")
            task["replicas_per_block"] = int(selected.get("required_replicas") or REPLICAS_PER_BLOCK)
            task["submissions_count"] = int(selected.get("submissions_count") or 0)
        task["trainer_backend"] = TRAINER_BACKEND
        if TRAINER_BACKEND == "llm":
            task["adapter_name"] = llm_adapter_name
            task["base_model"] = llm_base_model
            task["tokenizer"] = llm_tokenizer_name
    emit_event("info", "task_assigned", node_id=node_id, block_id=task.get("block_id"), model_version=task.get("model_version"))
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


@app.route("/events", methods=["GET"])
def events():
    client_q: queue.Queue = queue.Queue(maxsize=200)
    with _event_clients_lock:
        _event_clients.append(client_q)
        backlog = list(_event_buffer)

    def gen():
        try:
            for evt in backlog:
                yield f"data: {json.dumps(evt)}\n\n"
            while True:
                evt = client_q.get()
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            with _event_clients_lock:
                if client_q in _event_clients:
                    _event_clients.remove(client_q)

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


@app.route("/node_heartbeat", methods=["POST"])
def node_heartbeat():
    payload = request.get_json(force=True, silent=True) or {}
    node_id = payload.get("node_id") or request.headers.get("X-Node-Id") or request.remote_addr or "unknown"
    state = payload.get("state", "idle")
    capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {}
    upsert_node(TRAINER_DB_PATH, node_id=node_id, state=state, last_seen=_now(), enabled=bool(payload.get("enabled", True)), capabilities=capabilities)
    return jsonify({"ok": True}), 200


@app.route("/worker_status", methods=["GET"])
def worker_status():
    with state_lock:
        state = "training" if aggregator.pending_updates.get(aggregator.current_version) else "idle"
        blocks = []
        for entry in recent_history(TRAINER_DB_PATH, limit=10):
            blocks.append(
                {
                    "block_id": entry.get("block_id", ""),
                    "status": entry.get("status", ""),
                    "loss": entry.get("loss", ""),
                    "updated_at": entry.get("updated_at", ""),
                    "node_id": entry.get("node_id", ""),
                }
            )
        status = {
            "node_id": os.getenv("NODE_ID", "trainer"),
            "state": state,
            "base_model_version": aggregator.current_version if aggregator else None,
            "adapter_name": llm_adapter_name if TRAINER_BACKEND == "llm" else None,
            "quantization": None,
            "use_flash_attn": False,
            "compile_model": False,
            "blocks_processed": len(blocks),
            "blocks": blocks,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "trainer_backend": TRAINER_BACKEND,
            "base_model": llm_base_model if TRAINER_BACKEND == "llm" else None,
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
    losses = []
    max_steps = 20

    if TRAINER_BACKEND != "llm":
        # Fast MLP-only placeholder.
        tmp_model = MLPPolicy(obs_dim=32, action_dim=8).to(device)
        opt = torch.optim.Adam(tmp_model.parameters(), lr=1e-3)
        step = 0
        for f in files:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if step >= max_steps:
                        break
                    try:
                        obj = json.loads(line)
                        text = (obj.get("input", "") + " " + obj.get("target", "")).strip()
                    except Exception:
                        text = ""
                    feat = len(text) % 100
                    obs = torch.tensor([[feat] + [0] * 31], dtype=torch.float32, device=device)
                    logits, _ = tmp_model(obs)
                    target = torch.tensor([feat % 8], dtype=torch.long, device=device)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
                    step += 1
        elapsed = time.time() - start
        avg_loss = sum(losses) / len(losses) if losses else float("nan")
        return (
            jsonify({"ok": True, "message": f"starter_test(mlp) steps={len(losses)}, avg_loss={avg_loss:.4f}, time={elapsed:.2f}s"}),
            200,
        )

    # LLM backend: do a tiny LoRA step over starter-block pairs.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name or llm_base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=1e-4)
    model.train()
    step = 0
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                if step >= max_steps:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt = str(obj.get("input", "")).rstrip() + "\n"
                target = str(obj.get("target", "")).strip()
                if not prompt.strip() or not target:
                    continue
                prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=256).input_ids
                target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=256).input_ids
                input_ids = (prompt_ids + target_ids)[:256]
                labels = ([-100] * len(prompt_ids) + target_ids)[:256]
                input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
                labels_t = torch.tensor([labels], dtype=torch.long, device=device)
                attn = torch.ones_like(input_ids_t)
                out = model(input_ids=input_ids_t, attention_mask=attn, labels=labels_t)
                loss = out.loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                opt.step()
                losses.append(float(loss.item()))
                step += 1
    model.eval()
    elapsed = time.time() - start
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    return (
        jsonify({"ok": True, "message": f"starter_test(llm) steps={len(losses)}, avg_loss={avg_loss:.4f}, time={elapsed:.2f}s"}),
        200,
    )


@app.route("/enqueue_blocks", methods=["POST"])
def enqueue_blocks():
    payload = request.get_json(force=True, silent=True) or {}
    blocks_path = payload.get("blocks_path")
    dataset_label = payload.get("dataset_label", "unknown")
    target_adapter = payload.get("target_adapter")
    replicas = int(payload.get("replicas") or REPLICAS_PER_BLOCK)
    best_k = int(payload.get("best_k") or BEST_K_PER_BLOCK)
    if not blocks_path or not Path(blocks_path).exists():
        return jsonify({"error": "blocks_path not found"}), 400
    blocks: List[Dict] = []
    with Path(blocks_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not obj.get("block_id"):
                obj["block_id"] = str(uuid.uuid4())
            blocks.append(obj)
    added = db_enqueue_blocks(
        TRAINER_DB_PATH,
        blocks,
        dataset_label=dataset_label,
        target_adapter=target_adapter,
        updated_at=_now(),
        required_replicas=replicas,
        best_k=best_k,
    )
    emit_event(
        "info",
        "blocks_enqueued",
        count=added,
        dataset_label=dataset_label,
        target_adapter=target_adapter,
        replicas=replicas,
        best_k=best_k,
    )
    return jsonify({"message": f"Enqueued {added} blocks from {blocks_path}"}), 200


@app.route("/get_block", methods=["GET"])
def get_block():
    block_id = request.args.get("block_id")
    if not block_id:
        return jsonify({"error": "block_id required"}), 400
    block = db_get_block(TRAINER_DB_PATH, block_id)
    if block:
        return jsonify({"block": block}), 200
    return jsonify({"error": "block not found"}), 404


@app.route("/queue_status", methods=["GET"])
def queue_status():
    counts = queue_counts(TRAINER_DB_PATH)
    blocks = list_recent_blocks(TRAINER_DB_PATH, limit=50)
    return jsonify(
        {
            **counts,
            "blocks": blocks,
        }
    )


@app.route("/cluster_status", methods=["GET"])
def cluster_status():
    nodes = []
    for row in list_nodes(TRAINER_DB_PATH):
        nodes.append(
            {
                "node_id": row.get("node_id"),
                "state": row.get("state"),
                "blocks_completed": int(row.get("blocks_completed") or 0),
                "trust_score": float(row.get("trust_score") or 0.0),
                "last_seen": row.get("last_seen"),
                "enabled": bool(row.get("enabled")),
                "capabilities": row.get("capabilities", {}),
            }
        )
    return jsonify({"nodes": nodes})


@app.route("/disable_node", methods=["POST"])
def disable_node():
    payload = request.get_json(force=True, silent=True) or {}
    node_id = payload.get("node_id")
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    ok = set_node_enabled(TRAINER_DB_PATH, node_id=node_id, enabled=False, updated_at=_now())
    if not ok:
        return jsonify({"error": "node not found"}), 404
    emit_event("info", "node_disabled", node_id=node_id)
    return jsonify({"message": f"Node {node_id} disabled"}), 200


@app.route("/enable_node", methods=["POST"])
def enable_node():
    payload = request.get_json(force=True, silent=True) or {}
    node_id = payload.get("node_id")
    if not node_id:
        return jsonify({"error": "node_id required"}), 400
    ok = set_node_enabled(TRAINER_DB_PATH, node_id=node_id, enabled=True, updated_at=_now())
    if not ok:
        return jsonify({"error": "node not found"}), 404
    emit_event("info", "node_enabled", node_id=node_id)
    return jsonify({"message": f"Node {node_id} enabled"}), 200


@app.route("/export_adapter", methods=["POST"])
def export_adapter():
    if TRAINER_BACKEND != "llm":
        return jsonify({"ok": False, "error": "export_adapter only supported for TRAINER_BACKEND=llm"}), 400
    payload = request.get_json(force=True, silent=True) or {}
    adapter_name = payload.get("adapter_name") or llm_adapter_name
    if not adapter_name:
        return jsonify({"ok": False, "error": "adapter_name required"}), 400
    manifest_path = payload.get("manifest_path") or os.getenv("LLM_MANIFEST_PATH", "data/adapters/manifest.json")
    try:
        from src.models.adapter_manifest import select_adapter

        adapter_path, _entry = select_adapter(adapter_name, manifest_path)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"manifest lookup failed: {exc}"}), 400

    out_dir = Path(payload.get("output_dir") or adapter_path)
    # Safety: keep exports inside models/adapters unless explicitly allowed.
    allowed_root = Path("models/adapters").resolve()
    if allowed_root not in out_dir.resolve().parents and out_dir.resolve() != allowed_root:
        return jsonify({"ok": False, "error": f"output_dir must be under {allowed_root}"}), 400
    out_dir.mkdir(parents=True, exist_ok=True)
    with state_lock:
        model.save_pretrained(str(out_dir))
    (out_dir / "trainer_export.json").write_text(
        json.dumps(
            {
                "exported_at": _now(),
                "model_version": aggregator.current_version,
                "adapter_name": adapter_name,
                "base_model": llm_base_model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    emit_event("info", "adapter_exported", adapter_name=adapter_name, output_dir=str(out_dir), model_version=aggregator.current_version)
    return jsonify({"ok": True, "adapter_name": adapter_name, "output_dir": str(out_dir), "model_version": aggregator.current_version}), 200


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
    required = ["task_id", "base_model_version", "num_samples", "lora_delta", "assignment_id"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"accepted": False, "reason": f"missing fields: {', '.join(missing)}"}), 400

    base_version = int(payload["base_model_version"])
    num_samples = int(payload["num_samples"])
    metrics = payload.get("metrics", {})
    if not metrics_valid(metrics):
        return jsonify({"accepted": False, "reason": "metrics_invalid"}), 400

    assignment_id = str(payload.get("assignment_id"))
    assignment = get_assignment(TRAINER_DB_PATH, assignment_id)
    if not assignment:
        return jsonify({"accepted": False, "reason": "unknown_assignment"}), 400
    block_id = assignment.get("block_id")
    node_id = payload.get("node_id") or request.headers.get("X-Node-Id") or request.remote_addr or "unknown"
    if payload.get("block_id") and payload.get("block_id") != block_id:
        return jsonify({"accepted": False, "reason": "block_id_mismatch"}), 400
    if REQUIRE_BLOCK_HASH:
        if "block_id" not in payload or "block_hash" not in payload:
            return jsonify({"accepted": False, "reason": "block_hash_required"}), 400
        submitted_hash = payload["block_hash"]
        canonical_hash = block_hashes.get(block_id)
        if canonical_hash is None:
            node_block = db_get_block(TRAINER_DB_PATH, block_id)
            if node_block:
                canonical_hash = node_block.get("block_hash")
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
        # Persist raw submission and mark assignment complete.
        mark_assignment_status(TRAINER_DB_PATH, assignment_id=assignment_id, status="completed", updated_at=_now())
        record_submission(
            TRAINER_DB_PATH,
            assignment_id=assignment_id,
            block_id=block_id,
            node_id=node_id,
            base_version=base_version,
            num_samples=num_samples,
            delta_b64=payload["lora_delta"],
            metrics=metrics if isinstance(metrics, dict) else {},
            score=0.0,
            selected=False,
            created_at=_now(),
        )
        emit_event("info", "submission_received", node_id=node_id, block_id=block_id, assignment_id=assignment_id)

        # If we have enough replicas for this block, decide winners and submit one aggregated delta to the round aggregator.
        meta = get_block_meta(TRAINER_DB_PATH, block_id)
        required_replicas = int((meta or {}).get("required_replicas") or REPLICAS_PER_BLOCK)
        best_k = int((meta or {}).get("best_k") or BEST_K_PER_BLOCK)
        decided = bool((meta or {}).get("decided"))

        subs = list_block_submissions(TRAINER_DB_PATH, block_id)
        if (not decided) and len(subs) >= required_replicas:
            selected_ids, agg_delta, total_samples = _select_best_deltas(subs, best_k=best_k)
            if selected_ids and agg_delta:
                set_selected_submissions(TRAINER_DB_PATH, selected_ids, selected=True)
                accepted, aggregated_now = aggregator.submit_update(
                    base_version=base_version,
                    num_samples=total_samples,
                    delta_state=agg_delta,
                    metrics={"block_id": block_id, "selected_k": len(selected_ids)},
                )
                if not accepted:
                    emit_event("warn", "block_decision_stale", block_id=block_id, base_version=base_version)
                else:
                    set_block_decided(TRAINER_DB_PATH, block_id=block_id, decided=True, status="completed", updated_at=_now())
                    mark_block_status(TRAINER_DB_PATH, block_id=block_id, status="completed", updated_at=_now())
                    try:
                        increment_node_blocks_completed(TRAINER_DB_PATH, node_id=node_id, amount=1)
                    except Exception:
                        pass
                    loss_val = metrics.get("train_loss_mean", None) if isinstance(metrics, dict) else None
                    append_history(
                        TRAINER_DB_PATH,
                        block_id=block_id,
                        node_id=node_id,
                        status="decided",
                        loss=float(loss_val) if loss_val is not None else None,
                        updated_at=_now(),
                        model_version=aggregator.current_version,
                    )
                    emit_event(
                        "info",
                        "block_decided",
                        block_id=block_id,
                        required_replicas=required_replicas,
                        best_k=best_k,
                        selected_ids=selected_ids,
                        aggregated_now=aggregated_now,
                    )
                    if aggregated_now:
                        apply_agg_state_to_model()
                        if TRAINER_BACKEND == "llm":
                            save_lora_checkpoint(
                                lora_state=aggregator.get_current_state(),
                                version=aggregator.current_version,
                                output_dir=CHECKPOINT_DIR,
                                metadata={"adapter_name": llm_adapter_name, "base_model": llm_base_model},
                            )
                        else:
                            assert optimizer is not None
                            save_checkpoint(model, optimizer, aggregator.current_version, CHECKPOINT_DIR)

    return jsonify({"accepted": True, "current_policy_version": aggregator.current_version})


if __name__ == "__main__":
    init_trainer()
    app.run(host="0.0.0.0", port=5001)
