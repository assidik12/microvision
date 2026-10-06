import csv

import numpy as np

from microvision_attend.attendance import AttendanceWriter
from microvision_attend.matcher import EmbeddingRegistry


def test_registry_matches_normalized_embedding(tmp_path):
    registry = EmbeddingRegistry(["alice", "bob"], np.array([[1, 0], [0, 1]], dtype=np.float32))

    match = registry.match(np.array([2, 0], dtype=np.float32), threshold=0.8)

    assert match.name == "alice"
    assert match.score == 1.0


def test_registry_rejects_below_threshold():
    registry = EmbeddingRegistry(["alice"], np.array([[1, 0]], dtype=np.float32))

    match = registry.match(np.array([0, 1], dtype=np.float32), threshold=0.8)

    assert match.name is None
    assert match.score == 0.0


def test_attendance_cooldown(tmp_path):
    path = tmp_path / "attendance.csv"
    writer = AttendanceWriter(path, cooldown_seconds=30)

    assert writer.record("alice", now=100.0)
    assert not writer.record("alice", now=110.0)
    assert writer.record("alice", now=131.0)

    with path.open(newline="", encoding="utf-8") as file:
        assert len(list(csv.reader(file))) == 3
