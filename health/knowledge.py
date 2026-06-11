import asyncio
import json
from datetime import datetime
from typing import Dict, Optional

import httpx

from health.config import HealthConfig
from health.types import IntentResult, MedicalKnowledge


class MedicalKnowledgeClient:
    def __init__(self, config: HealthConfig, cache: Optional[Dict] = None):
        self.config = config
        self._cache: Dict[str, MedicalKnowledge] = cache or {}
        self._client: Optional[httpx.AsyncClient] = None

    async def query(self, intent_result: IntentResult) -> MedicalKnowledge:
        if not self.config.medical_api_url:
            return MedicalKnowledge(
                content="外部医学查询服务未配置。",
                source="unavailable"
            )

        cache_key = f"{intent_result.intent}:{':'.join(intent_result.entities)}"

        # L1: Call external API with retries
        api_result = await self._call_api_with_retry(intent_result)
        if api_result is not None:
            # Cache successful result
            if self.config.enable_knowledge_cache:
                self._cache[cache_key] = api_result
            return api_result

        # L2: Check cache
        cached = self._cache.get(cache_key)
        if cached is not None and self._is_cache_valid(cached):
            return cached

        # L3: LLM fallback (generate general knowledge from intent)
        return await self._llm_fallback(intent_result)

    async def _call_api_with_retry(self, intent_result: IntentResult) -> Optional[MedicalKnowledge]:
        payload = {
            "intent": intent_result.intent,
            "entities": intent_result.entities,
            "query": intent_result.suggested_query
        }
        headers = {}
        if self.config.medical_api_key:
            headers["Authorization"] = f"Bearer {self.config.medical_api_key}"

        for attempt in range(self.config.medical_api_max_retries + 1):
            try:
                response = await self._request(
                    method="POST",
                    url=self.config.medical_api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.medical_api_timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content") or data.get("result") or data.get("answer", "")
                    return MedicalKnowledge(
                        content=content,
                        source="api",
                        api_name=self.config.medical_api_url,
                        cached_at=datetime.now()
                    )
            except Exception:
                if attempt < self.config.medical_api_max_retries:
                    await asyncio.sleep(2 ** attempt)  # exponential backoff
                continue

        return None

    async def _request(self, **kwargs) -> httpx.Response:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return await self._client.request(**kwargs)

    def _is_cache_valid(self, knowledge: MedicalKnowledge) -> bool:
        if knowledge.cached_at is None:
            return False
        age = (datetime.now() - knowledge.cached_at).total_seconds()
        return age < self.config.cache_ttl_seconds

    async def _llm_fallback(self, intent_result: IntentResult) -> MedicalKnowledge:
        # L3 fallback: construct a general medical commonsense response
        # based on intent and entities, without calling any external API
        entities_text = "、".join(intent_result.entities) if intent_result.entities else "相关症状"

        if intent_result.intent == "symptom":
            content = (
                f"关于{entities_text}：常见原因包括疲劳、压力、睡眠不足、脱水或轻微感染。"
                f"如果症状持续或加重，建议咨询医生。"
            )
        elif intent_result.intent == "medication":
            content = (
                f"关于{entities_text}：任何药物都可能有副作用。"
                f"请仔细阅读药品说明书，并在医生指导下使用。"
            )
        elif intent_result.intent == "lifestyle":
            content = (
                f"关于{entities_text}：保持健康的生活方式有助于整体健康。"
                f"建议均衡饮食、规律运动和充足睡眠。"
            )
        else:
            content = (
                f"关于{entities_text}：请保持健康的生活习惯。"
                f"如有不适，建议及时就医。"
            )

        return MedicalKnowledge(
            content=content,
            source="llm_fallback"
        )

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
