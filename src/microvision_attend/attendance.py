from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path


class AttendanceWriter:
    """Append-only CSV writer with an in-memory cooldown to prevent duplicates."""

    def __init__(self, path: Path, cooldown_seconds: float) -> None:
        self._path = path
        self._cooldown_seconds = cooldown_seconds
        self._last_seen: dict[str, float] = {}

    def record(self, name: str, *, now: float | None = None) -> bool:
        timestamp = time.time() if now is None else now
        previous = self._last_seen.get(name)
        if previous is not None and timestamp - previous < self._cooldown_seconds:
            return False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists()
        with self._path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(["timestamp_utc", "name"])
            writer.writerow([datetime.now(timezone.utc).isoformat(), name])
        self._last_seen[name] = timestamp
        return True
