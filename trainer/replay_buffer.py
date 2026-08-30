from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import random


class ReplayBuffer:
    """
    Simple FIFO replay buffer for experience blocks.
    Stores tuples: (obs, action, reward, done, policy_version, info)
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def add(
        self,
        obs,
        action,
        reward: float,
        done: bool,
        policy_version: int,
        info: Optional[Dict] = None,
    ):
        self.buffer.append((obs, action, reward, done, policy_version, info or {}))

    def extend(self, items: List[Tuple]):
        for item in items:
            self.buffer.append(item)

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)
