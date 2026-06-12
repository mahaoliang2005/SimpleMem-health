import asyncio
import json
import re
from typing import List, Literal, Optional

from health.config import HealthConfig
from health.types import IntentResult
from utils.llm_client import LLMClient


class HealthIntentExtractor:
    def __init__(self, llm_client: LLMClient, config: Optional[HealthConfig] = None):
        self.llm_client = llm_client
        self.config = config or HealthConfig()
        self._urgent_pattern = re.compile(
            "|".join(re.escape(kw) for kw in self.config.urgent_keywords),
            re.IGNORECASE
        )

    async def extract(self, user_text: str) -> IntentResult:
        urgency = self._detect_urgency(user_text)
        try:
            intent_data = await self._extract_with_llm(user_text)
        except Exception:
            return IntentResult(
                intent="general",
                entities=[],
                urgency=urgency,
                suggested_query=user_text[:100]
            )
        if urgency == "urgent":
            intent_data["urgency"] = "urgent"
        return IntentResult(
            intent=intent_data.get("intent", "general"),
            entities=intent_data.get("entities", []),
            urgency=intent_data.get("urgency", "non-urgent"),
            suggested_query=intent_data.get("suggested_query", user_text[:100])
        )

    def _detect_urgency(self, user_text: str) -> Literal["urgent", "non-urgent"]:
        if self._urgent_pattern.search(user_text):
            return "urgent"
        return "non-urgent"

    async def _extract_with_llm(self, user_text: str) -> dict:
        prompt = self._build_prompt(user_text)
        messages = [
            {"role": "system", "content": "You are a health intent classification assistant. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        response = await asyncio.to_thread(
            self.llm_client.chat_completion,
            messages=messages,
            temperature=0.1
        )
        return json.loads(response)

    def _build_prompt(self, user_text: str) -> str:
        return f"""Analyze the following user message and classify its health-related intent.

User message: "{user_text}"

Return a JSON object with these fields:
- "intent": one of ["symptom", "medication", "vital", "lifestyle", "general"]
- "entities": list of health-related entities mentioned (symptoms, medications, body parts, etc.)
- "urgency": "urgent" or "non-urgent" (only "urgent" for life-threatening symptoms)
- "suggested_query": a query string optimized for semantic memory retrieval, combining the intent, entities, and related health keywords (in the same language as the user message)

Example:
Input: "我最近头痛得厉害，晚上也睡不着"
Output: {{
  "intent": "symptom",
  "entities": ["头痛", "失眠"],
  "urgency": "non-urgent",
  "suggested_query": "symptom 头痛 失眠 睡眠 健康"
}}
"""
