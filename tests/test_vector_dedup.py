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

    def test_fail_open_on_embedding_error(self):
        entries = [MemoryEntry(lossless_restatement="Test", keywords=["test"])]
        mock_emb = Mock()
        mock_emb.encode_documents.side_effect = RuntimeError("Embedding failed")
        result = deduplicate_entries(entries, mock_emb, threshold=0.85)
        # Should return original entries unchanged
        assert len(result) == 1
        assert result[0].superseded_by is None
