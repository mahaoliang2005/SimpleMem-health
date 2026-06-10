import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import Mock
from models.memory_entry import MemoryEntry
from core.vector_dedup import deduplicate_entries


class TestDeduplicateEntries:
    def test_empty_input(self):
        result = deduplicate_entries([], Mock(), threshold=0.85)
        assert result == []

    def test_single_entry_unchanged(self):
        entry = MemoryEntry(lossless_restatement="Alice likes coffee", keywords=["Alice", "coffee"])
        mock_emb = Mock()
        mock_emb.encode_documents.return_value = np.array([[1.0, 0.0, 0.0]])
        result = deduplicate_entries([entry], mock_emb, threshold=0.85)
        assert len(result) == 1
        assert result[0].superseded_by is None

    def test_merge_pair_above_threshold(self):
        entries = [
            MemoryEntry(lossless_restatement="Alice likes coffee", keywords=["Alice", "coffee"]),
            MemoryEntry(lossless_restatement="Alice enjoys coffee", keywords=["Alice", "coffee"]),
        ]
        mock_emb = Mock()
        # Both embeddings are identical -> sim = 1.0 > 0.85
        mock_emb.encode_documents.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        result = deduplicate_entries(entries, mock_emb, threshold=0.85, strategy="keep_longer")
        # First entry is shorter, so it should be marked as superseded
        assert result[0].superseded_by == entries[1].entry_id
        assert result[1].superseded_by is None

    def test_no_merge_below_threshold(self):
        entries = [
            MemoryEntry(lossless_restatement="Alice likes coffee", keywords=["Alice", "coffee"]),
            MemoryEntry(lossless_restatement="Bob hates tea", keywords=["Bob", "tea"]),
        ]
        mock_emb = Mock()
        # Orthogonal embeddings -> sim = 0.0 < 0.85
        mock_emb.encode_documents.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        result = deduplicate_entries(entries, mock_emb, threshold=0.85)
        assert all(e.superseded_by is None for e in result)

    def test_keep_longer_strategy(self):
        short = MemoryEntry(lossless_restatement="Short text", keywords=["a"])
        long = MemoryEntry(lossless_restatement="This is a much longer text with more detail", keywords=["a"])
        mock_emb = Mock()
        mock_emb.encode_documents.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        result = deduplicate_entries([short, long], mock_emb, threshold=0.85, strategy="keep_longer")
        assert result[0].superseded_by == long.entry_id
        assert result[1].superseded_by is None

    def test_keep_newer_strategy(self):
        old = MemoryEntry(
            lossless_restatement="Old text",
            keywords=["a"],
            timestamp="2025-01-01T10:00:00"
        )
        new = MemoryEntry(
            lossless_restatement="New text",
            keywords=["a"],
            timestamp="2025-01-02T10:00:00"
        )
        mock_emb = Mock()
        mock_emb.encode_documents.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        result = deduplicate_entries([old, new], mock_emb, threshold=0.85, strategy="keep_newer")
        assert result[0].superseded_by == new.entry_id
        assert result[1].superseded_by is None

    def test_transitive_chain_canonicalized(self):
        # A, B, C all identical -> B beats A, C beats B -> A should point directly to C
        a = MemoryEntry(lossless_restatement="A", keywords=["x"])
        b = MemoryEntry(lossless_restatement="B", keywords=["x"])
        c = MemoryEntry(lossless_restatement="C longer text", keywords=["x"])
        mock_emb = Mock()
        mock_emb.encode_documents.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        result = deduplicate_entries([a, b, c], mock_emb, threshold=0.85, strategy="keep_longer")
        # C is longest, so A and B should both point directly to C
        assert result[0].superseded_by == c.entry_id
        assert result[1].superseded_by == c.entry_id
        assert result[2].superseded_by is None

    def test_fail_open_on_embedding_error(self):
        entries = [MemoryEntry(lossless_restatement="Test", keywords=["test"])]
        mock_emb = Mock()
        mock_emb.encode_documents.side_effect = RuntimeError("Embedding failed")
        result = deduplicate_entries(entries, mock_emb, threshold=0.85)
        # Should return original entries unchanged
        assert len(result) == 1
        assert result[0].superseded_by is None

    def test_fail_open_on_similarity_error(self):
        entries = [MemoryEntry(lossless_restatement="Test", keywords=["test"])]
        mock_emb = Mock()
        mock_emb.encode_documents.return_value = np.array([[1.0, 0.0, 0.0]])
        import core.vector_dedup as vd
        original_norm = np.linalg.norm
        try:
            np.linalg.norm = Mock(side_effect=RuntimeError("norm failed"))
            result = deduplicate_entries(entries, mock_emb, threshold=0.85)
            assert len(result) == 1
            assert result[0].superseded_by is None
        finally:
            np.linalg.norm = original_norm
