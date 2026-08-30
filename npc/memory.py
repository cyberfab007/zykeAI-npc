from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class NPCEpisodeMemory:
    """
    Minimal memory container for an NPC agent loop.
    """

    owner_id: str = "player"
    recent_events: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)

    def add_event(self, text: str, *, max_events: int = 20) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.recent_events.append(text)
        if len(self.recent_events) > max_events:
            del self.recent_events[: len(self.recent_events) - max_events]

    def snapshot(self) -> Dict[str, object]:
        return {"owner_id": self.owner_id, "recent_events": list(self.recent_events), "goals": list(self.goals)}

    def as_context_text(self) -> str:
        parts = []
        if self.goals:
            parts.append("Goals: " + "; ".join(self.goals))
        if self.recent_events:
            parts.append("Recent events: " + " | ".join(self.recent_events[-10:]))
        return "\n".join(parts).strip()

