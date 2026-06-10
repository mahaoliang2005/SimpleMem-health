"""
Vector-based batch deduplication for memory entries.

Computes pairwise cosine similarity over a batch of entries,
marks near-duplicates with superseded_by (soft delete),
and returns the surviving entries.
"""
from typing import List, Literal, Optional
import numpy as np
from models.memory_entry import MemoryEntry


def deduplicate_entries(
    entries: List[MemoryEntry],
    embedding_model,
    threshold: float = 0.85,
    strategy: Literal["keep_longer", "keep_newer", "keep_both"] = "keep_longer"
) -> List[MemoryEntry]:
    """
    Deduplicate a batch of MemoryEntry objects using vector cosine similarity.

    Args:
        entries: List of MemoryEntry to deduplicate.
        embedding_model: Object with encode_documents(texts) -> np.ndarray method.
        threshold: Cosine similarity threshold. Pairs with sim > threshold are merged.
        strategy: How to choose the survivor — "keep_longer", "keep_newer", or "keep_both".

    Returns:
        The input list (modified in-place with superseded_by markers).
        Callers should write survivors and mark dropped entries in the store.
    """
    if not entries or strategy == "keep_both":
        return entries

    if len(entries) == 1:
        return entries

    try:
        restatements = [e.lossless_restatement for e in entries]
        vectors = embedding_model.encode_documents(restatements)
    except Exception as e:
        print(f"[Deduplication] Embedding failed: {e}. Skipping dedup.")
        return entries

    try:
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized = vectors / norms

        # Compute pairwise cosine similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)

        # Only consider upper triangle (excluding diagonal) to avoid duplicate pairs
        n = len(entries)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    winner, loser = _choose_winner(entries[i], entries[j], strategy)
                    if loser.superseded_by is None:
                        loser.superseded_by = winner.entry_id

        # Canonicalize superseded_by chains: ensure every loser points to the root winner
        entry_map = {e.entry_id: e for e in entries}
        for e in entries:
            if e.superseded_by is not None:
                current_id = e.superseded_by
                while current_id in entry_map and entry_map[current_id].superseded_by is not None:
                    current_id = entry_map[current_id].superseded_by
                e.superseded_by = current_id
    except Exception as e:
        print(f"[Deduplication] Similarity computation failed: {e}. Skipping dedup.")
        return entries

    return entries


def _choose_winner(
    a: MemoryEntry,
    b: MemoryEntry,
    strategy: Literal["keep_longer", "keep_newer", "keep_both"]
) -> tuple[MemoryEntry, MemoryEntry]:
    """Return (winner, loser) according to the chosen strategy."""
    if strategy == "keep_longer":
        len_a = len(a.lossless_restatement)
        len_b = len(b.lossless_restatement)
        if len_a > len_b:
            return a, b
        elif len_b > len_a:
            return b, a
        # Tie-break by lexicographically smaller entry_id for determinism
        return (a, b) if a.entry_id < b.entry_id else (b, a)

    elif strategy == "keep_newer":
        # None timestamps lose; otherwise compare isoformat strings
        if a.timestamp and not b.timestamp:
            return a, b
        elif b.timestamp and not a.timestamp:
            return b, a
        elif a.timestamp and b.timestamp:
            if a.timestamp > b.timestamp:
                return a, b
            elif b.timestamp > a.timestamp:
                return b, a
        # Tie (both None or equal) — break by entry_id
        return (a, b) if a.entry_id < b.entry_id else (b, a)

    # Fallback (should not reach here)
    return (a, b) if a.entry_id < b.entry_id else (b, a)
