#!/usr/bin/env python3
"""
Simple latency benchmark for the /generate endpoint across quantization, flash-attn, compile, and batch sizes.
Adjust BASE_URL, persona/context/state/player_input as needed.
"""
import statistics
import time
from typing import Any, Dict, List, Tuple

import requests

BASE_URL = "http://localhost:5000/generate"
RUNS_PER_CONFIG = 5

PERSONA = "npc_merchant"
CONTEXT = "Town market at noon."
STATE = "Player reputation: neutral."
PLAYER_INPUT = "Describe your wares to a new traveler in town."

CONFIGS: List[Tuple[str, str, bool, bool, int]] = [
    ("8bit_plain_b1", "8bit", False, False, 1),
    ("8bit_flash_b1", "8bit", True, False, 1),
    ("8bit_flash_comp_b1", "8bit", True, True, 1),
    ("4bit_plain_b1", "4bit", False, False, 1),
    ("4bit_flash_b1", "4bit", True, False, 1),
    ("4bit_flash_comp_b1", "4bit", True, True, 1),
    ("8bit_flash_b4", "8bit", True, False, 4),
    ("8bit_flash_comp_b4", "8bit", True, True, 4),
    ("4bit_flash_b4", "4bit", True, False, 4),
    ("4bit_flash_comp_b4", "4bit", True, True, 4),
]


def build_payload(quantization: str, use_flash_attn: bool, compile_model: bool, batch_size: int) -> Dict[str, Any]:
    if batch_size == 1:
        return {
            "persona": PERSONA,
            "context": CONTEXT,
            "state": STATE,
            "player_input": PLAYER_INPUT,
            "audience": "adult",  # no safety overhead for benchmarking
            "quantization": quantization,
            "use_flash_attn": use_flash_attn,
            "compile_model": compile_model,
        }
    return {
        "requests": [
            {
                "persona": PERSONA,
                "context": CONTEXT,
                "state": STATE,
                "player_input": PLAYER_INPUT,
                "audience": "adult",
            }
            for _ in range(batch_size)
        ],
        "quantization": quantization,
        "use_flash_attn": use_flash_attn,
        "compile_model": compile_model,
    }


def time_request(payload: Dict[str, Any]) -> float:
    t0 = time.time()
    resp = requests.post(BASE_URL, json=payload, timeout=120)
    t1 = time.time()
    resp.raise_for_status()
    return (t1 - t0) * 1000.0


def benchmark_config(name: str, quantization: str, use_flash_attn: bool, compile_model: bool, batch_size: int):
    print(f"\n=== {name} ===")
    payload = build_payload(quantization, use_flash_attn, compile_model, batch_size)

    # Warmup
    try:
        print("  Warmup...", end="", flush=True)
        _ = time_request(payload)
        print(" done.")
    except Exception as exc:
        print(f"\n  Warmup failed: {exc}")
        return None

    latencies = []
    for i in range(RUNS_PER_CONFIG):
        try:
            ms = time_request(payload)
            latencies.append(ms)
            print(f"  Run {i+1}/{RUNS_PER_CONFIG}: {ms:.1f} ms")
        except Exception as exc:
            print(f"  Run {i+1} failed: {exc}")
            break

    if not latencies:
        return None

    return {
        "name": name,
        "batch_size": batch_size,
        "quantization": quantization,
        "use_flash_attn": use_flash_attn,
        "compile_model": compile_model,
        "runs": len(latencies),
        "avg_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def main():
    results = []
    for name, quant, flash, comp, bs in CONFIGS:
        res = benchmark_config(name, quant, flash, comp, bs)
        if res:
            results.append(res)

    print("\n================ SUMMARY ================")
    header = f"{'name':24} {'bs':>3} {'q':>4} {'flash':>6} {'comp':>6} {'runs':>4} {'avg_ms':>8} {'p50_ms':>8} {'min_ms':>8} {'max_ms':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<24} "
            f"{r['batch_size']:>3} "
            f"{r['quantization']:>4} "
            f"{str(r['use_flash_attn']):>6} "
            f"{str(r['compile_model']):>6} "
            f"{r['runs']:>4} "
            f"{r['avg_ms']:>8.1f} "
            f"{r['p50_ms']:>8.1f} "
            f"{r['min_ms']:>8.1f} "
            f"{r['max_ms']:>8.1f}"
        )


if __name__ == "__main__":
    main()
