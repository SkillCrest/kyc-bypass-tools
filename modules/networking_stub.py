from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class StatusBroker:
    """Placeholder for future pub/sub logging or IPC."""

    subscribers: List[Callable[[str], None]]

    def publish(self, message: str) -> None:
        for s in self.subscribers:
            try:
                s(message)
            except Exception:
                pass

