import asyncio
from typing import List

from health.config import HealthConfig
from health.types import IntentResult, PersonalHealthContext
from main import SimpleMemSystem
from models.memory_entry import MemoryEntry


class HealthProfile:
    def __init__(self, simplemem: SimpleMemSystem, config: HealthConfig = None):
        self.simplemem = simplemem
        self.config = config or HealthConfig()

    async def retrieve_context(
        self,
        user_id: str,
        intent_result: IntentResult,
        top_k: int = None
    ) -> PersonalHealthContext:
        top_k = top_k or self.config.profile_top_k
        query = intent_result.suggested_query or " ".join(intent_result.entities)

        try:
            # SimpleMem is sync; bridge with asyncio.to_thread
            entries: List[MemoryEntry] = await asyncio.to_thread(
                self.simplemem.hybrid_retriever.retrieve,
                query
            )
        except Exception:
            # Fail-open: return empty context
            return PersonalHealthContext(entries=[], summary="")

        # Limit to top_k
        entries = entries[:top_k]

        # Build summary text for prompt injection
        summary = self._build_summary(entries)

        return PersonalHealthContext(entries=entries, summary=summary)

    def _build_summary(self, entries: List[MemoryEntry]) -> str:
        if not entries:
            return ""
        parts = []
        for e in entries:
            ts = f" ({e.timestamp})" if e.timestamp else ""
            parts.append(f"- {e.lossless_restatement}{ts}")
        return "\n".join(parts)
