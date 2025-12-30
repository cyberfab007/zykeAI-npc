import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import wraps
from pathlib import Path
from threading import Lock

from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generator import generate_npc_response

API_TOKEN = os.getenv("API_TOKEN")
REQUIRE_API_TOKEN = os.getenv("REQUIRE_API_TOKEN", "true").lower() == "true"
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SEC = float(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
METRICS_REQUIRE_AUTH = os.getenv("METRICS_REQUIRE_AUTH", "true").lower() == "true"
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
ALLOW_CUSTOM_ADAPTER_PATH = os.getenv("ALLOW_CUSTOM_ADAPTER_PATH", "false").lower() == "true"
DEFAULT_BASE_MODEL = os.getenv("DEFAULT_BASE_MODEL", "models/latest")
DEFAULT_TOKENIZER_PATH = os.getenv("DEFAULT_TOKENIZER_PATH", "models/latest")
DEFAULT_ADAPTER_NAME = os.getenv("DEFAULT_ADAPTER_NAME")
DEFAULT_MANIFEST_PATH = os.getenv("DEFAULT_MANIFEST_PATH", "data/adapters/manifest.json")

app = Flask(__name__)

requests_counter = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
latency_hist = Histogram("api_latency_seconds", "API latency in seconds", ["endpoint", "status"])

rate_lock = Lock()
rate_buckets = {}
executor = ThreadPoolExecutor(max_workers=8)
concurrency_lock = Lock()
concurrency_count = 0


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if REQUIRE_API_TOKEN:
            if not API_TOKEN:
                return jsonify({"error": "unauthorized: API token not configured"}), 401
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {API_TOKEN}":
                return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)

    return wrapper


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    if METRICS_REQUIRE_AUTH:
        if not API_TOKEN:
            return jsonify({"error": "unauthorized: API token not configured"}), 401
        auth_header = request.headers.get("Authorization", "")
        if auth_header != f"Bearer {API_TOKEN}":
            return jsonify({"error": "unauthorized"}), 401
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/generate", methods=["POST"])
@require_auth
def generate():
    start_total = time.time()
    endpoint = "generate"
    rate_key = request.headers.get("Authorization", "") or request.remote_addr or "anon"
    if not _check_rate_limit(rate_key):
        requests_counter.labels(endpoint=endpoint, status="rate_limited").inc()
        return jsonify({"error": "rate_limited"}), 429
    if not _acquire_concurrency():
        requests_counter.labels(endpoint=endpoint, status="concurrency_limited").inc()
        return jsonify({"error": "concurrency_limited"}), 429

    payload = request.get_json(force=True, silent=True) or {}

    # Support batching: payload may contain {"requests": [ ... ]} or a single request body.
    requests_body = payload.get("requests")
    if requests_body and isinstance(requests_body, list):
        shared_flags = {
            "quantization": payload.get("quantization"),
            "use_flash_attn": payload.get("use_flash_attn", False),
            "compile_model": payload.get("compile_model", False),
            "audience": payload.get("audience"),
            "base_model": payload.get("base_model"),
            "tokenizer_path": payload.get("tokenizer_path"),
            "adapter_path": payload.get("adapter_path"),
            "safe_mode": payload.get("safe_mode"),
            "enforce_schema": payload.get("enforce_schema"),
            "max_new_tokens": payload.get("max_new_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "top_k": payload.get("top_k"),
            "num_beams": payload.get("num_beams"),
        }
        results = []
        try:
            for req in requests_body:
                merged = dict(req)
                # Batch-level flags override per-request flags to keep configs consistent within a batch.
                for k, v in shared_flags.items():
                    if v is not None:
                        merged[k] = v
                res = _handle_single_request(merged)
                results.append(res)
        except Exception as exc:
            requests_counter.labels(endpoint=endpoint, status="error").inc()
            return jsonify({"error": str(exc)}), 500

        elapsed = time.time() - start_total
        app.logger.info(
            {
                "event": "inference",
                "batch_size": len(requests_body),
                "latency_ms": elapsed * 1000,
                "quantization": shared_flags.get("quantization"),
                "use_flash_attn": shared_flags.get("use_flash_attn"),
                "compile_model": shared_flags.get("compile_model"),
            }
        )
        latency_hist.labels(endpoint=endpoint, status="success").observe(elapsed)
        requests_counter.labels(endpoint=endpoint, status="success").inc()
        _release_concurrency()
        return jsonify({"results": results}), 200
    else:
        try:
            result = _handle_single_request(payload)
        except Exception as exc:
            requests_counter.labels(endpoint=endpoint, status="error").inc()
            _release_concurrency()
            return jsonify({"error": str(exc)}), 500
        elapsed = time.time() - start_total
        app.logger.info(
            {
                "event": "inference",
                "batch_size": 1,
                "latency_ms": elapsed * 1000,
                "quantization": payload.get("quantization"),
                "use_flash_attn": payload.get("use_flash_attn", False),
                "compile_model": payload.get("compile_model", False),
            }
        )
        latency_hist.labels(endpoint=endpoint, status="success").observe(elapsed)
        requests_counter.labels(endpoint=endpoint, status="success").inc()
        _release_concurrency()
        return jsonify(result), 200


def _handle_single_request(body: dict):
    required = ["persona", "context", "state", "player_input"]
    missing = [k for k in required if not body.get(k)]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    audience = (body.get("audience") or "minor").lower()
    npc_type = body.get("npc_type", "generic")
    base_model = body.get("base_model", DEFAULT_BASE_MODEL)
    tokenizer_path = body.get("tokenizer_path", DEFAULT_TOKENIZER_PATH)
    adapter_path = body.get("adapter_path")  # optional
    adapter_name = body.get("adapter_name") or DEFAULT_ADAPTER_NAME  # optional (manifest lookup)
    manifest_path = body.get("manifest_path", DEFAULT_MANIFEST_PATH)
    adapter_version = body.get("adapter_version")
    if adapter_path and not ALLOW_CUSTOM_ADAPTER_PATH:
        raise ValueError("adapter_path not allowed; use adapter_name from manifest or enable ALLOW_CUSTOM_ADAPTER_PATH")
    max_new_tokens = int(body.get("max_new_tokens", 80))
    # Defaults: minors => safety on + schema enforced; adults => safety off + no retries/guards
    default_safe = False if audience == "adult" else True
    safe_mode = bool(body.get("safe_mode", default_safe))
    enforce_schema = bool(body.get("enforce_schema", audience != "adult"))
    quantization = body.get("quantization")  # None, "4bit", or "8bit"
    temperature = float(body.get("temperature", 0.8))
    top_p = float(body.get("top_p", 0.9))
    top_k = body.get("top_k")
    top_k = int(top_k) if top_k is not None else None
    num_beams = body.get("num_beams")
    num_beams = int(num_beams) if num_beams is not None else None
    use_flash_attn = bool(body.get("use_flash_attn", False))
    compile_model = bool(body.get("compile_model", False))
    enable_tools = bool(body.get("enable_tools", False))
    tool_manifest_path = body.get("tool_manifest_path", "data/mcp/tools.json")
    mcp_timeout_sec = float(body.get("mcp_timeout_sec", 5.0))
    max_tool_calls = int(body.get("max_tool_calls", 2))

    def task():
        return generate_npc_response(
            base_model=base_model,
            tokenizer_path=tokenizer_path,
        adapter_path=adapter_path,
        adapter_name=adapter_name,
        manifest_path=manifest_path,
        persona=body["persona"],
        context=body["context"],
        state=body["state"],
        player_input=body["player_input"],
            max_new_tokens=max_new_tokens,
            safe_mode=safe_mode,
            quantization=quantization,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            enforce_schema=enforce_schema,
            use_flash_attn=use_flash_attn,
            compile_model=compile_model,
            cache_tag=f"{adapter_name}:{adapter_version}" if (adapter_name and adapter_version is not None) else None,
            enable_tools=enable_tools,
            npc_type=npc_type,
            audience=audience,
            tool_manifest_path=tool_manifest_path,
            mcp_timeout_sec=mcp_timeout_sec,
            max_tool_calls=max_tool_calls,
        )

    try:
        future = executor.submit(task)
        result = future.result(timeout=REQUEST_TIMEOUT_SEC)
    except FuturesTimeout:
        future.cancel()
        requests_counter.labels(endpoint="generate", status="timeout").inc()
        raise TimeoutError(f"request exceeded {REQUEST_TIMEOUT_SEC}s")

    return {"result": result, "audience": audience}


def _check_rate_limit(key: str) -> bool:
    now = time.time()
    with rate_lock:
        window_start = now - RATE_LIMIT_WINDOW_SEC
        bucket = rate_buckets.get(key, [])
        bucket = [ts for ts in bucket if ts >= window_start]
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            rate_buckets[key] = bucket
            return False
        bucket.append(now)
        rate_buckets[key] = bucket
        return True


def _acquire_concurrency() -> bool:
    global concurrency_count
    with concurrency_lock:
        if concurrency_count >= MAX_CONCURRENCY:
            return False
        concurrency_count += 1
        return True


def _release_concurrency() -> None:
    global concurrency_count
    with concurrency_lock:
        concurrency_count = max(0, concurrency_count - 1)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
