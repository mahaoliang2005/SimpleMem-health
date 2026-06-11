import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from health.knowledge import MedicalKnowledgeClient
from health.types import IntentResult, MedicalKnowledge
from health.config import HealthConfig


@pytest.fixture
def config():
    return HealthConfig(
        medical_api_url="https://api.example.com/medical",
        medical_api_key="test-key",
        medical_api_timeout=5.0,
        medical_api_max_retries=2,
        enable_knowledge_cache=True
    )


@pytest.fixture
def client(config):
    return MedicalKnowledgeClient(config=config)


class TestMedicalKnowledgeClient:
    @pytest.mark.asyncio
    async def test_api_success(self, client):
        intent = IntentResult(intent="symptom", entities=["头痛"])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "头痛常见原因包括睡眠不足、压力等"}
        mock_response.text = '{"content": "头痛常见原因包括睡眠不足、压力等"}'

        with patch("httpx.AsyncClient.request", return_value=mock_response):
            result = await client.query(intent)

        assert result.source == "api"
        assert "头痛" in result.content

    @pytest.mark.asyncio
    async def test_api_l1_retry_then_success(self, client):
        intent = IntentResult(intent="symptom", entities=["头痛"])
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.text = "error"

        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"content": "ok"}
        mock_success.text = '{"content": "ok"}'

        with patch("httpx.AsyncClient.request", side_effect=[mock_fail, mock_success]):
            result = await client.query(intent)

        assert result.source == "api"
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_api_l2_cache_fallback(self, client):
        # Pre-populate cache
        client._cache["symptom:头痛"] = MedicalKnowledge(
            content="cached content",
            source="cache",
            cached_at=datetime.now() - timedelta(minutes=5)
        )
        intent = IntentResult(intent="symptom", entities=["头痛"])

        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.text = "error"

        with patch("httpx.AsyncClient.request", return_value=mock_fail):
            result = await client.query(intent)

        assert result.source == "cache"
        assert result.content == "cached content"

    @pytest.mark.asyncio
    async def test_api_l3_llm_fallback(self, client):
        intent = IntentResult(intent="symptom", entities=["头痛"])
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.text = "error"

        with patch("httpx.AsyncClient.request", return_value=mock_fail):
            result = await client.query(intent)

        assert result.source == "llm_fallback"
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_no_api_configured(self, client):
        client.config.medical_api_url = None
        intent = IntentResult(intent="symptom", entities=["头痛"])
        result = await client.query(intent)
        assert result.source == "unavailable"
