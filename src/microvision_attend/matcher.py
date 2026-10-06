from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Match:
    name: str | None
    score: float


class EmbeddingRegistry:
    """Small in-memory registry; embeddings are normalized once at load time."""

    def __init__(self, names: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(names):
            raise ValueError("names and embeddings must contain the same number of rows")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embeddings must not contain zero vectors")
        self._names = names
        self._embeddings = embeddings / norms

    @classmethod
    def load(cls, path: Path) -> EmbeddingRegistry:
        if not path.exists():
            return cls([], np.empty((0, 0), dtype=np.float32))
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = [str(item["name"]) for item in payload]
        vectors = np.asarray([item["embedding"] for item in payload], dtype=np.float32)
        if not names:
            vectors = np.empty((0, 0), dtype=np.float32)
        return cls(names, vectors)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"name": name, "embedding": vector.tolist()}
            for name, vector in zip(self._names, self._embeddings, strict=True)
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")

    def match(self, embedding: np.ndarray, threshold: float) -> Match:
        if not self._names:
            return Match(None, 0.0)
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return Match(None, 0.0)
        scores = self._embeddings @ (vector / norm)
        index = int(np.argmax(scores))
        score = float(scores[index])
        return Match(self._names[index], score) if score >= threshold else Match(None, score)
