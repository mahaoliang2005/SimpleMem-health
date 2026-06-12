import pytest

from health.prompt_builder import HealthPromptBuilder
from health.types import IntentResult, MedicalKnowledge, PersonalHealthContext
from models.memory_entry import MemoryEntry


class TestHealthPromptBuilder:
    def test_build_with_all_sources(self):
        builder = HealthPromptBuilder()
        personal = PersonalHealthContext(
            entries=[MemoryEntry(lossless_restatement="User has insomnia history")],
            summary="- User has insomnia history"
        )
        knowledge = MedicalKnowledge(
            content="Headache common causes: stress, lack of sleep",
            source="api"
        )
        intent = IntentResult(intent="symptom", entities=["headache"])

        messages = builder.build("Why do I have a headache?", personal, knowledge, intent)

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "health assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "headache" in messages[1]["content"]
        assert "insomnia" in messages[1]["content"]
        assert "stress" in messages[1]["content"]

    def test_build_with_llm_fallback_source(self):
        builder = HealthPromptBuilder()
        personal = PersonalHealthContext()
        knowledge = MedicalKnowledge(
            content="General info about headaches",
            source="llm_fallback"
        )
        intent = IntentResult(intent="symptom", entities=["headache"])

        messages = builder.build("Why do I have a headache?", personal, knowledge, intent)

        user_msg = messages[1]["content"]
        assert "[Medical knowledge service unavailable" in user_msg

    def test_build_with_unavailable_source(self):
        builder = HealthPromptBuilder()
        personal = PersonalHealthContext()
        knowledge = MedicalKnowledge(
            content="",
            source="unavailable"
        )
        intent = IntentResult(intent="symptom", entities=["headache"])

        messages = builder.build("Why do I have a headache?", personal, knowledge, intent)

        user_msg = messages[1]["content"]
        assert "[External medical query service temporarily unavailable" in user_msg

    def test_build_empty_personal_context(self):
        builder = HealthPromptBuilder()
        personal = PersonalHealthContext()
        knowledge = MedicalKnowledge(content="Some info", source="api")
        intent = IntentResult(intent="general")

        messages = builder.build("Hello", personal, knowledge, intent)

        user_msg = messages[1]["content"]
        assert "No relevant personal health history" in user_msg
