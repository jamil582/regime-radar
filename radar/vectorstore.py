"""In-memory vector index built around Embed 5's Matryoshka + quantisation features.

Storage per document:
  * `full`   – int8-quantised 2048-d vector (4x smaller than float32) used for exact rescoring
  * `coarse` – the first `coarse_dim` dims (Matryoshka prefix), re-normalised, packed to *binary*
               (1 bit/dim) for a Hamming-distance first pass

Search = binary Hamming shortlist over the coarse prefix  ->  int8 cosine rescoring on full vectors.
For a 2048-d model that is a 256-bit (32-byte) fingerprint per document for the first pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def quantize_int8(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row int8 quantisation. Returns (int8 matrix, per-row scale)."""
    scale = np.abs(mat).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(mat / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def matryoshka_prefix(mat: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to the first `dim` components and re-normalise (valid for Matryoshka-trained models)."""
    pre = mat[:, :dim].astype(np.float32)
    n = np.linalg.norm(pre, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pre / n


def pack_binary(mat: np.ndarray) -> np.ndarray:
    return np.packbits(mat > 0, axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(query_bits: np.ndarray, doc_bits: np.ndarray) -> np.ndarray:
    return _POPCOUNT[np.bitwise_xor(doc_bits, query_bits)].sum(axis=1)


@dataclass
class Hit:
    doc: dict[str, Any]
    score: float
    coarse_rank: int


class VectorIndex:
    def __init__(self, name: str, coarse_dim: int = 256):
        self.name = name
        self.coarse_dim = coarse_dim
        self.docs: list[dict[str, Any]] = []
        self._q: np.ndarray | None = None       # int8 full vectors
        self._scale: np.ndarray | None = None   # per-row scales
        self._bits: np.ndarray | None = None    # packed binary coarse vectors
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self.docs)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._ids

    def add(self, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if not docs:
            return
        assert len(docs) == len(vectors)
        q, s = quantize_int8(vectors)
        bits = pack_binary(matryoshka_prefix(vectors, self.coarse_dim))
        if self._q is None:
            self._q, self._scale, self._bits = q, s, bits
        else:
            self._q = np.vstack([self._q, q])
            self._scale = np.vstack([self._scale, s])
            self._bits = np.vstack([self._bits, bits])
        self.docs.extend(docs)
        self._ids.update(d["id"] for d in docs)

    def full_vectors(self) -> np.ndarray:
        if self._q is None:
            return np.zeros((0, 0), dtype=np.float32)
        return self._q.astype(np.float32) * self._scale

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 8,
        shortlist: int = 64,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[Hit]:
        if self._q is None or not len(self.docs):
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        candidates = np.arange(len(self.docs))
        if where is not None:
            candidates = np.array([i for i in candidates if where(self.docs[i])], dtype=int)
            if not len(candidates):
                return []
        # Stage 1: Hamming distance on the binary Matryoshka prefix.
        qbits = pack_binary(matryoshka_prefix(q, self.coarse_dim))
        dist = hamming(qbits, self._bits[candidates])
        order = np.argsort(dist, kind="stable")[: max(shortlist, k)]
        short = candidates[order]
        # Stage 2: exact cosine on de-quantised full vectors.
        full = self._q[short].astype(np.float32) * self._scale[short]
        full /= np.linalg.norm(full, axis=1, keepdims=True) + 1e-9
        scores = full @ q[0]
        best = np.argsort(-scores)[:k]
        return [Hit(self.docs[short[i]], float(scores[i]), int(i)) for i in best]

    def exact_search(self, query_vec: np.ndarray, k: int = 8) -> list[Hit]:
        """Brute-force float baseline, used by the retrieval eval."""
        full = self.full_vectors()
        if not len(full):
            return []
        full /= np.linalg.norm(full, axis=1, keepdims=True) + 1e-9
        scores = full @ query_vec.astype(np.float32)
        best = np.argsort(-scores)[:k]
        return [Hit(self.docs[i], float(scores[i]), -1) for i in best]

    def stats(self) -> dict[str, Any]:
        n = len(self.docs)
        dim = 0 if self._q is None else self._q.shape[1]
        return {
            "name": self.name,
            "docs": n,
            "dim": dim,
            "coarse_dim": self.coarse_dim,
            "bytes_int8": n * dim,
            "bytes_binary_prefix": n * (self.coarse_dim // 8),
            "bytes_float32_equiv": n * dim * 4,
        }

    # ------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        if self._q is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), q=self._q, scale=self._scale, bits=self._bits)
        path.with_suffix(".json").write_text(json.dumps(self.docs))

    def load(self, path: Path) -> bool:
        npz, js = path.with_suffix(".npz"), path.with_suffix(".json")
        if not (npz.exists() and js.exists()):
            return False
        data = np.load(npz)
        self._q, self._scale, self._bits = data["q"], data["scale"], data["bits"]
        self.docs = json.loads(js.read_text())
        self._ids = {d["id"] for d in self.docs}
        return True
