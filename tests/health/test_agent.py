import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from health.agent import HealthAgent
from health.types import HealthResponse, IntentResult, MedicalKnowledge, PersonalHealthContext


@pytest.fixture
def mock_simplemem():
    sm = MagicMock()
    sm.hybrid_retriever = MagicMock()
    return sm


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def agent(mock_simplemem, mock_llm):
    from health.config import HealthConfig
    from health.intent import HealthIntentExtractor
    from health.profile import HealthProfile
    from health.knowledge import MedicalKnowledgeClient
    from health.data_input import HealthDataInput
    from health.prompt_builder import HealthPromptBuilder

    config = HealthConfig()
    agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm, config=config)
    return agent


class TestHealthAgentAsk:
    @pytest.mark.asyncio
    async def test_ask_normal_flow(self, agent, mock_llm):
        # Mock intent extraction
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="symptom", entities=["头痛"], urgency="non-urgent",
            suggested_query="symptom 头痛"
        ))
        # Mock profile retrieval
        agent.profile.retrieve_context = AsyncMock(return_value=PersonalHealthContext(
            entries=[], summary=""
        ))
        # Mock knowledge query
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="头痛常见原因", source="api"
        ))
        # Mock LLM
        mock_llm.chat_completion.return_value = "你可能需要多休息。"

        result = await agent.ask("我最近头痛")

        assert isinstance(result, HealthResponse)
        assert "休息" in result.answer
        assert "api" in result.sources
        # Verify conversation archived
        agent.simplemem.add_dialogue.assert_called()

    @pytest.mark.asyncio
    async def test_ask_urgent_bypass(self, agent, mock_llm):
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="symptom", entities=["胸痛"], urgency="urgent"
        ))
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="", source="unavailable"
        ))
        result = await agent.ask("胸口很痛呼吸困难")

        assert "紧急" in result.answer or "URGENT" in result.answer
        assert result.latency_ms == 0
        assert result.sources == ["emergency_fallback"]
        mock_llm.chat_completion.assert_not_called()
        agent.knowledge_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_api_degraded(self, agent, mock_llm):
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="symptom", entities=["头痛"], urgency="non-urgent"
        ))
        agent.profile.retrieve_context = AsyncMock(return_value=PersonalHealthContext())
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="一般信息", source="llm_fallback"
        ))
        mock_llm.chat_completion.return_value = "建议多休息。"

        result = await agent.ask("头痛")
        assert "休息" in result.answer

    @pytest.mark.asyncio
    async def test_ask_llm_failure(self, agent, mock_llm):
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="general", urgency="non-urgent"
        ))
        agent.profile.retrieve_context = AsyncMock(return_value=PersonalHealthContext())
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="", source="unavailable"
        ))
        mock_llm.chat_completion.side_effect = Exception("LLM down")

        result = await agent.ask("你好")
        assert "暂时繁忙" in result.answer


class TestHealthAgentRecordVital:
    @pytest.mark.asyncio
    async def test_record_vital_success(self, agent):
        agent.data_input.record = AsyncMock(return_value={"status": "ok"})
        result = await agent.record_vital("user1", "blood_pressure", {"systolic": 120, "diastolic": 80})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_record_vital_validation_failure(self, agent):
        agent.data_input.record = AsyncMock(return_value={"status": "error", "message": "out of range"})
        result = await agent.record_vital("user1", "blood_pressure", {"systolic": 300})
        assert result["status"] == "error"
