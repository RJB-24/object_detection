"""Persistent known-face store: name -> list of SFace embeddings."""
from __future__ import annotations

import pickle
import threading
from pathlib import Path

import numpy as np

from app import config
from app.detectors.face import FaceEngine


class FaceDatabase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else Path(config.FACE_DB_PATH)
        self._lock = threading.Lock()
        self._data: dict[str, list[np.ndarray]] = {}
        self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> dict[str, list[np.ndarray]]:
        with self._lock:
            if self.path.exists():
                try:
                    with open(self.path, "rb") as fh:
                        raw = pickle.load(fh)
                    self._data = {
                        str(k): [np.asarray(e, dtype=np.float32) for e in v]
                        for k, v in dict(raw).items()
                    }
                except Exception:  # noqa: BLE001
                    self._data = {}
            else:
                self._data = {}
            return {k: list(v) for k, v in self._data.items()}

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({k: list(v) for k, v in self._data.items()}, fh)
        tmp.replace(self.path)

    # -- CRUD ----------------------------------------------------------
    def add_embedding(self, name: str, embedding: np.ndarray) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Name must not be empty")
        with self._lock:
            bucket = self._data.setdefault(name, [])
            bucket.append(np.asarray(embedding, dtype=np.float32).flatten())
            self._save_unlocked()
            return len(bucket)

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._data:
                del self._data[name]
                self._save_unlocked()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._save_unlocked()

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._data.keys())

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._data.items()}

    def snapshot(self) -> dict[str, list[np.ndarray]]:
        with self._lock:
            return {k: list(v) for k, v in self._data.items()}

    def total_embeddings(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._data.values())

    # -- matching ------------------------------------------------------
    def best_match(
        self, embedding: np.ndarray, threshold: float | None = None
    ) -> tuple[str | None, float, bool]:
        """Return (name, best_score, matched). Cosine similarity, higher = better."""
        thr = config.FACE_MATCH_THRESH if threshold is None else float(threshold)
        emb = np.asarray(embedding, dtype=np.float32).flatten()
        best_name: str | None = None
        best_score = -1.0
        with self._lock:
            items = list(self._data.items())
        for name, embs in items:
            for ref in embs:
                s = FaceEngine.cosine_similarity(emb, ref)
                if s > best_score:
                    best_score = s
                    best_name = name
        if best_name is None:
            return None, 0.0, False
        return best_name, float(best_score), bool(best_score >= thr)
