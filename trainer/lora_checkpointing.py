from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def latest_lora_checkpoint_path(checkpoint_dir: str) -> Optional[Path]:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("lora_v*.pt"))
    return candidates[-1] if candidates else None


def save_lora_checkpoint(
    lora_state: Dict[str, torch.Tensor],
    version: int,
    output_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a LoRA trainable-only state dict checkpoint.

    This avoids serializing the full base model weights.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"lora_v{version}.pt")
    state_cpu = {k: v.detach().to("cpu").clone() for k, v in lora_state.items()}
    torch.save({"version": version, "lora_state": state_cpu, "metadata": metadata or {}}, ckpt_path)
    return ckpt_path


def load_latest_lora_checkpoint(checkpoint_dir: str) -> Optional[Tuple[int, Dict[str, torch.Tensor], Dict[str, Any]]]:
    ckpt = latest_lora_checkpoint_path(checkpoint_dir)
    if ckpt is None:
        return None
    payload = torch.load(ckpt, map_location="cpu")
    version = int(payload.get("version", 0))
    lora_state = payload.get("lora_state") or {}
    if not isinstance(lora_state, dict):
        lora_state = {}
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return version, lora_state, metadata

