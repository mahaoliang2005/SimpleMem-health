import pytest
from unittest.mock import MagicMock, patch

from health.agent import HealthAgent
from health.config import HealthConfig
from health.types import HealthResponse, IntentResult
from models.memory_entry import MemoryEntry


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_ask_pipeline(self):
        """End-to-end: question → intent → parallel retrieval → answer → archive."""
        mock_simplemem = MagicMock()
        mock_simplemem.hybrid_retriever.retrieve.return_value = [
            MemoryEntry(lossless_restatement="User has a history of migraines")
        ]
        mock_llm = MagicMock()

        # First call: intent extraction
        # Second call: answer generation
        mock_llm.chat_completion.side_effect = [
            '{"intent": "symptom", "entities": ["头痛"], "urgency": "non-urgent", "suggested_query": "symptom 头痛"}',
            '根据你的历史，你之前有偏头痛记录。建议多休息。'
        ]

        agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm)

        with patch.object(agent.knowledge_client, "_call_api_with_retry", return_value=None):
            result = await agent.ask("我最近头痛")

        assert isinstance(result, HealthResponse)
        assert len(result.answer) > 0
        assert mock_simplemem.add_dialogue.call_count >= 2  # user + assistant archived

    @pytest.mark.asyncio
    async def test_record_then_retrieve(self):
        """Record a vital → ask about it → should retrieve the record."""
        mock_simplemem = MagicMock()
        mock_simplemem.hybrid_retriever.retrieve.return_value = [
            MemoryEntry(lossless_restatement="User user1 recorded blood pressure: 120/80 mmHg.")
        ]
        mock_llm = MagicMock()
        mock_llm.chat_completion.side_effect = [
            '{"intent": "vital", "entities": ["血压"], "urgency": "non-urgent", "suggested_query": "vital 血压"}',
            '你最近的血压是 120/80 mmHg，属于正常范围。'
        ]

        agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm)

        # Record BP
        record_result = await agent.record_vital(
            "user1", "blood_pressure", {"systolic": 120, "diastolic": 80}
        )
        assert record_result["status"] == "ok"

        # Ask about it
        with patch.object(agent.knowledge_client, "_call_api_with_retry", return_value=None):
            answer = await agent.ask("我最近血压怎么样？")

        assert "120/80" in answer.answer or "正常" in answer.answer

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Verify profile + knowledge retrieval happen in parallel."""
        import asyncio
        mock_simplemem = MagicMock()
        mock_simplemem.hybrid_retriever.retrieve.return_value = []
        mock_llm = MagicMock()
        mock_llm.chat_completion.return_value = '{"intent": "general", "entities": [], "urgency": "non-urgent", "suggested_query": "hello"}'

        agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm)

        call_order = []

        async def slow_profile(*args, **kwargs):
            await asyncio.sleep(0.05)
            call_order.append("profile")
            from health.types import PersonalHealthContext
            return PersonalHealthContext()

        async def slow_knowledge(*args, **kwargs):
            await asyncio.sleep(0.05)
            call_order.append("knowledge")
            from health.types import MedicalKnowledge
            return MedicalKnowledge(content="", source="unavailable")

        agent.profile.retrieve_context = slow_profile
        agent.knowledge_client.query = slow_knowledge

        mock_llm.chat_completion.return_value = "Hello!"

        start = asyncio.get_event_loop().time()
        result = await agent.ask("你好")
        elapsed = asyncio.get_event_loop().time() - start

        # Should be < 0.1s (parallel) not > 0.1s (sequential)
        assert elapsed < 0.15, f"Expected parallel execution, took {elapsed}s"
        assert "profile" in call_order
        assert "knowledge" in call_order
