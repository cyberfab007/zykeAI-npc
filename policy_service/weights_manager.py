import os
from pathlib import Path
from typing import Optional

import torch


class WeightsManager:
    """
    Polls a checkpoint directory for the latest policy_version and loads weights.
    """

    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.policy_version: int = -1
        self.model: Optional[torch.nn.Module] = None

    def latest_checkpoint(self) -> Optional[Path]:
        if not self.checkpoint_dir.exists():
            return None
        candidates = sorted(self.checkpoint_dir.glob("model_v*.pt"))
        return candidates[-1] if candidates else None

    def load_latest(self, make_model_fn):
        ckpt = self.latest_checkpoint()
        if not ckpt:
            return False
        state = torch.load(ckpt, map_location="cpu")
        version = int(state.get("policy_version", -1))
        if version <= self.policy_version:
            return False
        model = make_model_fn()
        model.load_state_dict(state["model_state"])
        model.eval()
        self.model = model
        self.policy_version = version
        return True
