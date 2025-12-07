import os
import time
from functools import wraps

from flask import Flask, jsonify, request

from src.inference.generator import generate_npc_response

API_TOKEN = os.getenv("API_TOKEN")  # Optional; set to require auth

app = Flask(__name__)

metrics = {
    "requests_total": 0,
    "requests_failed": 0,
    "latency_ms": [],
}


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if API_TOKEN:
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
    # Return basic counters; omit raw latencies for brevity
    lat = metrics["latency_ms"]
    summary = {
        "requests_total": metrics["requests_total"],
        "requests_failed": metrics["requests_failed"],
    }
    if lat:
        summary.update(
            {
                "latency_ms_p50": sorted(lat)[len(lat) // 2],
                "latency_ms_avg": sum(lat) / len(lat),
            }
        )
    return jsonify(summary), 200


@app.route("/generate", methods=["POST"])
@require_auth
def generate():
    metrics["requests_total"] += 1
    payload = request.get_json(force=True, silent=True) or {}
    required = ["persona", "context", "state", "player_input"]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        metrics["requests_failed"] += 1
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    base_model = payload.get("base_model", "models/latest")
    tokenizer_path = payload.get("tokenizer_path", "models/latest")
    adapter_path = payload.get("adapter_path")  # optional
    max_new_tokens = int(payload.get("max_new_tokens", 80))
    safe_mode = bool(payload.get("safe_mode", True))
    quantization = payload.get("quantization")  # None, "4bit", or "8bit"
    temperature = float(payload.get("temperature", 0.8))
    top_p = float(payload.get("top_p", 0.9))
    top_k = payload.get("top_k")
    top_k = int(top_k) if top_k is not None else None
    num_beams = payload.get("num_beams")
    num_beams = int(num_beams) if num_beams is not None else None

    start = time.time()
    try:
        result = generate_npc_response(
            base_model=base_model,
            tokenizer_path=tokenizer_path,
            adapter_path=adapter_path,
            persona=payload["persona"],
            context=payload["context"],
            state=payload["state"],
            player_input=payload["player_input"],
            max_new_tokens=max_new_tokens,
            safe_mode=safe_mode,
            quantization=quantization,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )
    except Exception as exc:
        metrics["requests_failed"] += 1
        return jsonify({"error": str(exc)}), 500
    finally:
        metrics["latency_ms"].append((time.time() - start) * 1000)

    return jsonify({"result": result}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
