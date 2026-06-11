import pytest
from unittest.mock import MagicMock

from health.profile import HealthProfile
from health.types import IntentResult, PersonalHealthContext
from models.memory_entry import MemoryEntry


@pytest.fixture
def mock_simplemem():
    sm = MagicMock()
    sm.hybrid_retriever = MagicMock()
    return sm


@pytest.fixture
def profile(mock_simplemem):
    return HealthProfile(simplemem=mock_simplemem)


class TestHealthProfile:
    @pytest.mark.asyncio
    async def test_retrieve_context_basic(self, profile, mock_simplemem):
        mock_entry = MemoryEntry(
            lossless_restatement="User reported headache on June 10",
            keywords=["headache", "user"]
        )
        mock_simplemem.hybrid_retriever.retrieve.return_value = [mock_entry]

        intent = IntentResult(
            intent="symptom",
            entities=["headache"],
            suggested_query="symptom headache"
        )
        result = await profile.retrieve_context("user1", intent)

        assert isinstance(result, PersonalHealthContext)
        assert len(result.entries) == 1
        assert "headache" in result.entries[0].lossless_restatement
        mock_simplemem.hybrid_retriever.retrieve.assert_called_once_with("symptom headache")

    @pytest.mark.asyncio
    async def test_retrieve_context_empty(self, profile, mock_simplemem):
        mock_simplemem.hybrid_retriever.retrieve.return_value = []

        intent = IntentResult(intent="general", suggested_query="hello")
        result = await profile.retrieve_context("user1", intent)

        assert result.entries == []
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_retrieve_context_limit_top_k(self, profile, mock_simplemem):
        mock_entries = [MemoryEntry(lossless_restatement=f"entry {i}") for i in range(20)]
        mock_simplemem.hybrid_retriever.retrieve.return_value = mock_entries

        intent = IntentResult(intent="symptom", suggested_query="test")
        result = await profile.retrieve_context("user1", intent, top_k=5)

        assert len(result.entries) == 5
