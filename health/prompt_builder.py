from typing import List

from health.types import IntentResult, MedicalKnowledge, PersonalHealthContext


class HealthPromptBuilder:
    SYSTEM_PROMPT: str = """You are a personal health assistant.

Important constraints:
1. You do NOT provide medical diagnoses or prescribe treatments.
2. If the user's symptoms sound serious, strongly recommend consulting a healthcare professional.
3. Prioritize referencing the user's personal health history when relevant to their question.
4. Be empathetic, clear, and concise.
"""

    def build(
        self,
        user_question: str,
        personal_context: PersonalHealthContext,
        medical_knowledge: MedicalKnowledge,
        intent: IntentResult,
    ) -> List[dict]:
        """Build the final LLM prompt messages list."""
        # Source attribution annotation
        source_note = ""
        if medical_knowledge.source == "llm_fallback":
            source_note = "\n[Medical knowledge service unavailable; the following is based on general medical commonsense.]"
        elif medical_knowledge.source == "unavailable":
            source_note = "\n[External medical query service temporarily unavailable.]"

        # Personal context section
        personal_section = ""
        if personal_context.summary:
            personal_section = f"""\n
[User's Personal Health History]
{personal_context.summary}"""
        else:
            personal_section = "\n\n[User's Personal Health History]\nNo relevant personal health history found."

        # Medical knowledge section
        knowledge_section = ""
        if medical_knowledge.content:
            knowledge_section = f"""\n
[Medical Knowledge]
{medical_knowledge.content}{source_note}"""
        elif medical_knowledge.source == "unavailable":
            knowledge_section = f"""\n
[Medical Knowledge]
{source_note}"""

        # User question section
        user_section = f"""\n
[User Question]
{user_question}

Please answer based on the user's personal health history and medical knowledge above."""

        user_content = f"{personal_section}{knowledge_section}{user_section}"

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
