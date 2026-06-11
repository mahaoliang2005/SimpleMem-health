import asyncio
import time
from typing import Optional

from health.config import HealthConfig
from health.data_input import HealthDataInput
from health.emergency import EmergencyResponse
from health.intent import HealthIntentExtractor
from health.knowledge import MedicalKnowledgeClient
from health.profile import HealthProfile
from health.prompt_builder import HealthPromptBuilder
from health.types import HealthResponse
from main import SimpleMemSystem
from utils.llm_client import LLMClient


class HealthAgent:
    def __init__(
        self,
        simplemem: SimpleMemSystem,
        llm_client: LLMClient,
        config: Optional[HealthConfig] = None
    ):
        self.simplemem = simplemem
        self.llm_client = llm_client
        self.config = config or HealthConfig()

        # Initialize submodules
        self.intent_extractor = HealthIntentExtractor(llm_client=llm_client, config=self.config)
        self.profile = HealthProfile(simplemem=simplemem, config=self.config)
        self.knowledge_client = MedicalKnowledgeClient(config=self.config)
        self.data_input = HealthDataInput(simplemem=simplemem)
        self.prompt_builder = HealthPromptBuilder()

    async def ask(self, user_question: str, user_id: str = "default") -> HealthResponse:
        start_time = time.time()

        # Step 1: Extract intent
        intent_result = await self.intent_extractor.extract(user_question)

        # Emergency fast-path
        if intent_result.urgency == "urgent":
            return EmergencyResponse.standard_reply(intent_result.entities)

        # Step 2: Parallel retrieval
        try:
            profile_task = asyncio.create_task(
                self.profile.retrieve_context(user_id, intent_result)
            )
            knowledge_task = asyncio.create_task(
                self.knowledge_client.query(intent_result)
            )
            personal_context, medical_knowledge = await asyncio.gather(
                profile_task, knowledge_task
            )
        except Exception:
            # If parallel execution fails, fall back to sequential with defaults
            personal_context = await self.profile.retrieve_context(user_id, intent_result)
            medical_knowledge = await self.knowledge_client.query(intent_result)

        # Step 3: Build prompt
        messages = self.prompt_builder.build(
            user_question=user_question,
            personal_context=personal_context,
            medical_knowledge=medical_knowledge,
            intent=intent_result,
        )

        # Step 4: Generate answer
        try:
            import asyncio as _asyncio
            answer = await _asyncio.to_thread(
                self.llm_client.chat_completion,
                messages=messages,
                temperature=0.3
            )
            sources = [medical_knowledge.source]
        except Exception:
            answer = "服务暂时繁忙，请稍后再试。"
            sources = ["fallback"]

        latency_ms = int((time.time() - start_time) * 1000)

        # Step 5: Archive conversation
        self._archive_conversation(user_question, answer)

        return HealthResponse(answer=answer, sources=sources, latency_ms=latency_ms)

    async def record_vital(
        self, user_id: str, sign_type: str, data: dict, timestamp: Optional[str] = None
    ) -> dict:
        return await self.data_input.record(user_id, sign_type, data, timestamp)

    def get_health_profile(self, user_id: str = "default"):
        # TODO: implement profile aggregation in future iteration
        return {"user_id": user_id, "status": "not yet implemented"}

    def _archive_conversation(self, user_question: str, answer: str):
        """Archive the Q&A pair to SimpleMem for future retrieval."""
        try:
            self.simplemem.add_dialogue(speaker="user", content=user_question)
            self.simplemem.add_dialogue(speaker="assistant", content=answer)
        except Exception:
            # Fail-open: log and continue if archiving fails
            pass

    async def close(self):
        await self.knowledge_client.close()
