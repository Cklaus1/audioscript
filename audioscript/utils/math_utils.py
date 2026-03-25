"""Shared math utilities."""

from __future__ import annotations

import math
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Pure Python implementation — works without numpy.
    Returns 0.0 for zero-length or zero-norm vectors.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_cosine_best_match(
    query: list[float],
    candidates: dict[str, list[float]],
    threshold: float = 0.0,
) -> tuple[str | None, float]:
    """Find the best cosine match for a query against multiple candidates.

    Uses numpy when available for batch computation (faster at 50+ candidates),
    falls back to pure Python.

    Args:
        query: The embedding to match.
        candidates: Dict of {id: embedding}.
        threshold: Minimum score to return a match.

    Returns:
        (best_id, best_score) or (None, 0.0) if no match above threshold.
    """
    if not candidates or not query:
        return None, 0.0

    try:
        import numpy as np

        ids = list(candidates.keys())
        matrix = np.array([candidates[k] for k in ids])
        q = np.array(query)

        # Batch cosine similarity via matrix multiply
        norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None, 0.0

        # Avoid division by zero
        valid = norms > 0
        scores = np.zeros(len(ids))
        scores[valid] = (matrix[valid] @ q) / (norms[valid] * q_norm)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return ids[best_idx], best_score
        return None, best_score

    except ImportError:
        # Fallback: iterate with pure Python
        best_id = None
        best_score = 0.0
        for cid, embedding in candidates.items():
            score = cosine_similarity(query, embedding)
            if score > best_score:
                best_score = score
                best_id = cid

        if best_id and best_score >= threshold:
            return best_id, best_score
        return None, best_score
