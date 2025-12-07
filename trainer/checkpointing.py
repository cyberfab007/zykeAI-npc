import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_version: int,
    output_dir: str,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"model_v{policy_version}.pt")
    torch.save(
        {
            "policy_version": policy_version,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_path: str,
) -> int:
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return int(state.get("policy_version", 0))
