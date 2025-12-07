import time
import uuid
from typing import Dict, List

import requests


class ExperienceLogger:
    def __init__(self, trainer_url: str, block_size: int = 256):
        self.trainer_url = trainer_url.rstrip("/")
        self.block_size = block_size
        self.buffer: List[Dict] = []
        self.policy_version = -1
        self.env_id = "default"
        self.npc_type = "generic"

    def log_step(self, obs, action, reward, done, policy_version: int, npc_type: str, env_id: str):
        self.policy_version = policy_version
        self.npc_type = npc_type
        self.env_id = env_id
        self.buffer.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
            }
        )
        if done or len(self.buffer) >= self.block_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        block = {
            "block_id": str(uuid.uuid4()),
            "policy_version": self.policy_version,
            "env_id": self.env_id,
            "npc_type": self.npc_type,
            "steps": self.buffer,
            "meta": {
                "timestamp_start": int(time.time() * 1000),
                "timestamp_end": int(time.time() * 1000),
            },
        }
        try:
            requests.post(f"{self.trainer_url}/api/experience_block", json=block, timeout=5)
        except Exception:
            pass
        self.buffer = []
