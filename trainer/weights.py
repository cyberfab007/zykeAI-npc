import os
from pathlib import Path
from typing import Optional

import torch


def latest_checkpoint_path(checkpoint_dir: str) -> Optional[Path]:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("model_v*.pt"))
    return candidates[-1] if candidates else None


def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_dir: str,
) -> Optional[int]:
    ckpt = latest_checkpoint_path(checkpoint_dir)
    if ckpt is None:
        return None
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return int(state.get("policy_version", 0))
