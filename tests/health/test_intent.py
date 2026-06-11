import pytest
from unittest.mock import MagicMock

from health.intent import HealthIntentExtractor
from health.types import IntentResult


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


@pytest.fixture
def extractor(mock_llm):
    return HealthIntentExtractor(llm_client=mock_llm)


class TestHealthIntentExtractor:
    @pytest.mark.asyncio
    async def test_intent_symptom(self, extractor, mock_llm):
        mock_llm.chat_completion.return_value = (
            '{"intent": "symptom", "entities": ["头痛"], '
            '"urgency": "non-urgent", "suggested_query": "symptom 头痛 健康 用药 睡眠"}'
        )
        result = await extractor.extract("我最近头痛")
        assert result.intent == "symptom"
        assert "头痛" in result.entities
        assert result.urgency == "non-urgent"
        assert "symptom" in result.suggested_query

    @pytest.mark.asyncio
    async def test_intent_medication(self, extractor, mock_llm):
        mock_llm.chat_completion.return_value = (
            '{"intent": "medication", "entities": ["阿司匹林"], '
            '"urgency": "non-urgent", "suggested_query": "medication 阿司匹林"}'
        )
        result = await extractor.extract("阿司匹林有什么副作用")
        assert result.intent == "medication"
        assert "阿司匹林" in result.entities

    @pytest.mark.asyncio
    async def test_intent_urgent(self, extractor, mock_llm):
        mock_llm.chat_completion.return_value = (
            '{"intent": "symptom", "entities": ["胸痛"], '
            '"urgency": "non-urgent", "suggested_query": "symptom 胸痛"}'
        )
        result = await extractor.extract("胸口很痛呼吸困难")
        assert result.urgency == "urgent"

    @pytest.mark.asyncio
    async def test_intent_general(self, extractor, mock_llm):
        mock_llm.chat_completion.return_value = "not valid json"
        result = await extractor.extract("你好")
        assert result.intent == "general"
        assert result.urgency == "non-urgent"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, extractor, mock_llm):
        mock_llm.chat_completion.side_effect = Exception("LLM error")
        result = await extractor.extract("whatever")
        assert result.intent == "general"
        assert result.urgency == "non-urgent"
