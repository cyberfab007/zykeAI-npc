# ZykeAI Prototype Ichoronium Galactic Space Networks NPC AI 

Scripts and scaffolding to fine-tune modern models with **Hugging Face Transformers**, optional **LoRA adapters**, and deploy behind a simple HTTP API.

Recommended baselines:
- **Open + small (plug-and-play):** `EleutherAI/pythia-410m-deduped` (or `EleutherAI/pythia-1.4b-deduped`)
- **Bigger (requires access):** `meta-llama/Llama-2-13b-hf` (higher quality, heavier)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Distributed Training Smoke Test (LoRA Delta Path)](#distributed-training-smoke-test-lora-delta-path)
- [Deployment API](#deployment-api)
- [Command Station (GUI)](#command-station-gui)
- [Notes](#notes)

---

## Project Structure

```text
.
├── configs/                     # YAML training/eval hyperparameters
├── data/                        # Data prep + dataset utilities
│   ├── raw/                     # (gitignored)
│   ├── processed/               # (gitignored)
│   └── lib/                     # Dataset utilities (e.g., Wikipedia downloader)
├── scripts/                     # Runnable entrypoints
│   ├── train.py
│   ├── evaluate.py
│   ├── command_station.py
│   ├── distributed_train.py
│   ├── hyperparameter_tuning.py
│   └── train_json.py
├── src/                         # Importable library code
│   ├── models/                  # Base model/tokenizer loading
│   ├── inference/               # Generation helpers
│   ├── cc/                      # Command-and-control server
│   ├── node/                    # Worker client
│   └── utils/                   # Config helpers
├── deployment/                  # Flask API server + Dockerfile
├── trainer/                     # Distributed training scaffold (trainer loop)
├── policy_service/              # Policy service stub
├── actors/                      # Actor client / worker / logger
├── npc/                         # NPC agent loops (sense→think→act)
├── examples/                    # Educational transformer implementation
├── openweb/                     # Open WebUI + Ollama compose
├── results/                     # (gitignored) training artifacts
├── models/                      # (gitignored) checkpoints + final weights + adapters
└── logs/                        # (gitignored) runs / metrics
```

---

## Quickstart

### Install deps (Python 3.12 recommended)

```bash
pip install -r requirements.txt
```

### (Optional) Link to external storage

Recommended for large artifacts (datasets, checkpoints, logs):

```bash
python data/link_external.py --external-root /mnt/SSD1TB/ZYKE_DATA
```

This links `data/raw`, `data/processed`, `models`, `results`, `logs` → your mount.

### Build a cleaned dataset (streaming + weighted)

```bash
python data/build_dataset.py \
  --sources openwebtext,wikipedia \
  --output-dir data/processed \
  --max-total 100000 \
  --weights openwebtext=1,wikipedia=1 \
  --local-dir data/raw/ebooks \
  --lang en \
  --use-minhash \
  --minhash-threshold 0.8
```

### (Optional) Custom prep

```bash
python data/prepare_data.py
```

### (Optional) Build experience blocks (for the distributed trainer)

```bash
python data/make_experience_blocks.py \
  --input-path data/processed \
  --output data/processed/experience_blocks.jsonl \
  --tokenizer EleutherAI/pythia-410m-deduped \
  --seq-len 128 \
  --steps-per-block 64 \
  --env-id ebooks \
  --npc-type generic
```

### (Optional) Smoke-test with tiny starter blocks

```bash
python scripts/train.py --starter-blocks data/starter_blocks --resume-latest --max-steps 50
```

### Train

Train on a text corpus (from `data/build_dataset.py`):

```bash
python scripts/train.py \
  --base-model EleutherAI/pythia-410m-deduped \
  --train-txt data/processed/train.txt \
  --val-txt data/processed/val.txt \
  --resume-latest
```

### LoRA adapter training

```bash
python scripts/train.py \
  --base-model EleutherAI/pythia-410m-deduped \
  --starter-blocks data/starter_blocks \
  --use-lora \
  --adapter-name npc_core_pythia_410m_v1 \
  --update-manifest \
  --resume-latest
```

Configure targets via `--lora-target-modules`.

### Logging & Hub publishing

- TensorBoard: `--log-to tensorboard` (logs under `logs/training`)
- Weights & Biases: `--log-to wandb` (requires `WANDB_PROJECT` + token)
- Hugging Face Hub push: `--hf-push`

> Note: LLaMA weights require Hugging Face access + token.

### Evaluate

```bash
python scripts/evaluate.py \
  --evals wikitext2,wikitext103,c4,lambada,piqa,hellaswag,winogrande,arc
```

### Evaluate an adapter

```bash
python scripts/evaluate.py \
  --evals wikitext2,lambada \
  --adapter-path models/adapters/<name> \
  --base-model meta-llama/Llama-2-13b-hf
```

Or use:

```bash
python scripts/evaluate.py \
  --evals wikitext2,lambada \
  --adapter-name <name>
```

`--adapter-name` is resolved via `data/adapters/manifest.json`.

### Use an adapter at inference

Python:

```python
from src.inference.generator import generate_npc_response

generate_npc_response(
  ...,
  adapter_path="models/adapters/<name>",
  base_model="meta-llama/Llama-2-13b-hf",
)
```

Or use `adapter_name` (resolved via `data/adapters/manifest.json`).

Important: keep the base model consistent with what the adapter was trained on.

### NPC schema inference

```bash
python -m src.inference.generator
```

- Safe-mode is on by default; set `safe_mode=False` for raw output.
- Optional quantization: `quantization="4bit"` or `"8bit"`.

### Prepare NPC schema dataset

```bash
python data/prepare_npc_schema.py \
  --input data/npc_sample.jsonl \
  --output-dir data/processed
```

### Deployment API (Flask)

```bash
API_TOKEN=yourtoken python deployment/app.py
```

`/generate` supports:
- `adapter_name` (preferred; resolved via `data/adapters/manifest.json`)
- `adapter_version` (optional cache-buster; use this when you update adapter weights on disk and want inference to reload)
 - MCP tools (optional): set `enable_tools=true` and pass `npc_type` so the server can allowlist tools from `data/mcp/tools.json`

### Distributed delta loop (trainer + workers)

Trainer:

```bash
python -m trainer.server
```

Trainer persistence + ops:
- SQLite DB: `TRAINER_DB_PATH=models/trainer.db`
- Live logs (SSE): `GET /events`

Worker(s):

```bash
python -m actors.worker --trainer-url http://localhost:5001 --node-id local
```

To run the **LLM LoRA-delta** backend (small baseline recommended):

```bash
TRAINER_BACKEND=llm LLM_ADAPTER_NAME=npc_core_pythia_410m_v1 python -m trainer.server
python -m actors.worker --trainer-url http://localhost:5001 --node-id local --mode llm --adapter-name npc_core_pythia_410m_v1
```

To enqueue blocks (generated by `data/make_experience_blocks.py`) into the trainer queue:

```bash
curl -X POST http://localhost:5001/enqueue_blocks \
  -H 'Content-Type: application/json' \
  -d '{"dataset_label":"local","blocks_path":"data/processed/experience_blocks.jsonl","target_adapter":"npc_core_pythia_410m_v1","replicas":5,"best_k":1}'
```

### Command Station (GUI)

```bash
python scripts/command_station.py
```

The GUI can:
- Start/stop the local worker (`mlp` or `llm` mode)
- Build blocks locally and enqueue them to the trainer
- Stream trainer logs live from `GET /events`
- Chat with the inference API; optionally calls `POST /export_adapter` on the trainer first and passes `adapter_version` to force inference reload
- (Optional) Tool-use planning: set `enable_tools` in your `/generate` payload and run a local MCP tool server (see below)

### MCP Tool Server (optional, for metaverse/game integration)

Run the local stub server (replace tool handlers with real game state later):

```bash
python -m src.mcp.server_stub
```

Tool manifest lives at `data/mcp/tools.json`. Add/allowlist tools by `npc_type` (e.g., `dog_guard`).

### Adapter manifest + publish

Manifest path:

```text
data/adapters/manifest.json
```

Publish adapter:

```bash
python scripts/publish_adapter.py \
  --adapter-name <name> \
  --manifest data/adapters/manifest.json \
  --repo-id <user/repo>
```

### Merge LoRA into a base model

```bash
python scripts/merge_adapters.py \
  --base-model ... \
  --adapter-path ... \
  --output models/merged/<name>
```

### PYTHONPATH for local scripts

If you import from `src/` in your own scripts, set:

```bash
export PYTHONPATH=.
```

---

## Distributed Training Smoke Test (LoRA Delta Path)

### 1) Start trainer service

Runs aggregation + timeout ticker:

```bash
python -m trainer.server
```

Environment knobs (env vars):

- `NUM_TASKS_PER_ROUND` (default `3`)
- `MIN_UPDATES_PER_ROUND` (default `1`)
- `ROUND_TIMEOUT_SEC` (default `30`)
- `MAX_STALENESS` (default `1`)
- `DELTA_NORM_MAX` (default `1e9`)
- `TICK_INTERVAL_SEC` (default `1`)
- `CHECKPOINT_DIR` (default `models/checkpoints`)

### 2) Start one or more workers

Workers compute real deltas and submit updates:

```bash
python actors/worker.py --trainer-url http://localhost:5001 --num-tasks 3
```

Worker flow:

1. `GET /get_task`
2. `GET /get_lora_weights`
3. Local train loop
4. Save fp16 delta (`torch.save`)
5. `POST /submit_update` with metrics:
   - `train_loss_mean/last`
   - `grad_norm_mean`
   - `steps`
   - `duration`
   - `num_samples`

### 3) What to expect

- After `NUM_TASKS_PER_ROUND` updates (or timeout + `MIN_UPDATES_PER_ROUND`), aggregation runs.
- `policy_version` increments.
- A new checkpoint is saved under `models/checkpoints`.

### Standalone base training (no distributed updates)

```bash
python trainer/trainer_loop.py
```

This can run standalone PPO on locally pulled experience blocks (polling endpoint placeholder in args).

---

## Deployment API

Run:

```bash
API_TOKEN=yourtoken python deployment/app.py
```

### Endpoints

- `GET /health`
- `GET /metrics`
- `POST /generate`

### `/generate` request schema

JSON fields:

- Core: `persona`, `context`, `state`, `player_input`
- Optional model selection: `adapter_path`, `adapter_name`, `base_model`
- Generation settings: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `num_beams`
- Behavior: `safe_mode`, `quantization`
- Batching:

```json
{
  "requests": [
    {
      "persona": "...",
      "context": "...",
      "state": "...",
      "player_input": "..."
    }
  ]
}
```

Batch-level flags apply to the whole batch.

---

## Notes

- License: **CC BY-NC 4.0** (non-commercial). See `LICENSE`.
- Large datasets/checkpoints/logs are gitignored; use external storage for big artifacts.
- Deployment is a simple Flask example; swap to FastAPI (or your stack) as needed.

### Logging / monitoring

- `--log-to tensorboard` → logs under `logs/training`
- `--log-to wandb` → requires `WANDB_PROJECT` + token
- Hub publish with `--hf-push`

### Adapter instructions

- Train with `--use-lora`, choose a name via `--adapter-name`
- Configure targets via `--lora-target-modules`
- Keep the base model consistent at train/eval/inference

### Inference options

- Safe-mode filtering for NPC JSON outputs (toggleable)
- Optional 4/8-bit quantization
- Adapter loading for modular skills

### Distributed delta safety

- Trainer rejects stale versions, bad shapes, NaN/inf metrics, and deltas above `DELTA_NORM_MAX`
- Ticker enforces `ROUND_TIMEOUT_SEC` aggregation

### Alignment / safety

- Generator retries JSON parsing with temperature/top-p backoff
- Enforces allowed enums
- `audience=minor` default keeps rails/schema; `audience=adult` bypasses them

Samples:

- `data/alignment/npc_alignment_sample.jsonl`
- `data/alignment/npc_alignment_dataset.jsonl`

### Inference efficiency & ops

- Optional flash-attention (`use_flash_attn`)
- `torch.compile` (`compile_model`)
- Quantization (4/8-bit)
- Per-process LRU caching
- Prometheus metrics at `/metrics`
- Rate limiting + timeouts
- Stricter auth default (`REQUIRE_API_TOKEN=true`)

### Adapter ecosystem

- Manifest at `data/adapters/manifest.json`
- Publish to Hub with `scripts/publish_adapter.py`
- Merge adapters with `scripts/merge_adapters.py`

### Deployment hardening

- Dockerfile uses `gunicorn` + healthcheck
- Configurable defaults via:
  - `DEFAULT_BASE_MODEL`
  - `DEFAULT_TOKENIZER_PATH`
  - `DEFAULT_ADAPTER_NAME`
  - `DEFAULT_MANIFEST_PATH`
- Concurrency caps and rate limiting enabled.
