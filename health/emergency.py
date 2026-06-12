from typing import List

from health.types import HealthResponse


class EmergencyResponse:
    """Immediate emergency replies. Zero latency, no LLM/API calls."""

    _STANDARD_REPLY: str = (
        "【紧急提醒】您的症状可能较为严重，请立即就医或拨打急救电话（中国大陆：120）。\n\n"
        "我作为健康助手无法提供紧急医疗诊断。请不要延误救治。"
    )

    _STANDARD_REPLY_EN: str = (
        "[URGENT] Your symptoms may be serious. Please seek emergency medical care immediately "
        "or call your local emergency number.\n\n"
        "I cannot provide emergency medical diagnosis. Do not delay seeking help."
    )

    @classmethod
    def standard_reply(cls, entities: List[str] = None) -> HealthResponse:
        entities = entities or []
        has_chinese = any("一" <= c <= "鿿" for e in entities for c in e)
        answer = cls._STANDARD_REPLY if has_chinese else cls._STANDARD_REPLY_EN
        return HealthResponse(
            answer=answer,
            sources=["emergency_fallback"],
            latency_ms=0
        )
