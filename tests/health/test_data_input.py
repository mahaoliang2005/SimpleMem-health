import pytest
from unittest.mock import MagicMock

from health.data_input import HealthDataInput
from health.validators import ValidationError


@pytest.fixture
def mock_simplemem():
    return MagicMock()


@pytest.fixture
def data_input(mock_simplemem):
    return HealthDataInput(simplemem=mock_simplemem)


class TestHealthDataInput:
    @pytest.mark.asyncio
    async def test_record_bp_success(self, data_input, mock_simplemem):
        result = await data_input.record(
            user_id="user1",
            sign_type="blood_pressure",
            data={"systolic": 120, "diastolic": 80}
        )
        assert result["status"] == "ok"
        mock_simplemem.add_dialogue.assert_called_once()
        call_args = mock_simplemem.add_dialogue.call_args
        assert "120" in call_args.kwargs["content"]
        assert "80" in call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_record_bp_invalid_blocked(self, data_input, mock_simplemem):
        result = await data_input.record(
            user_id="user1",
            sign_type="blood_pressure",
            data={"systolic": 300, "diastolic": 80}
        )
        assert result["status"] == "error"
        assert "out of range" in result["message"]
        mock_simplemem.add_dialogue.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_weight_success(self, data_input, mock_simplemem):
        result = await data_input.record(
            user_id="user1",
            sign_type="weight",
            data={"value": 70.5}
        )
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unknown_sign_type(self, data_input):
        result = await data_input.record(
            user_id="user1",
            sign_type="unknown",
            data={"value": 100}
        )
        assert result["status"] == "error"
        assert "Unknown" in result["message"]

    @pytest.mark.asyncio
    async def test_record_with_timestamp(self, data_input, mock_simplemem):
        ts = "2026-06-12T09:00:00"
        await data_input.record(
            user_id="user1",
            sign_type="heart_rate",
            data={"value": 72},
            timestamp=ts
        )
        call_args = mock_simplemem.add_dialogue.call_args
        assert call_args.kwargs["timestamp"] == ts
