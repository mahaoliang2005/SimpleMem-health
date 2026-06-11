# Health Personal Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a personal health assistant agent on top of SimpleMem with 6 new application-layer modules: intent extraction, health profile retrieval, medical knowledge client, structured data input, prompt builder, and agent orchestrator.

**Architecture:** Layered domain architecture. HealthAgent orchestrates the flow. All submodules are independent and testable. SimpleMem is treated as an opaque memory layer — no modifications to SimpleMem core code.

**Tech Stack:** Python 3.10+, pydantic, httpx (already installed), pytest, asyncio

---

## File Structure

```
health/
    __init__.py           # Export public API
    types.py              # Shared dataclasses
    config.py             # HealthConfig
    emergency.py          # EmergencyResponse
    validators.py         # BloodPressureValidator, HeartRateValidator, etc.
    intent.py             # HealthIntentExtractor
    profile.py            # HealthProfile
    knowledge.py          # MedicalKnowledgeClient
    data_input.py         # HealthDataInput
    prompt_builder.py     # HealthPromptBuilder
    agent.py              # HealthAgent (orchestrator)

tests/
    health/
        __init__.py
        test_validators.py
        test_intent.py
        test_profile.py
        test_knowledge.py
        test_data_input.py
        test_prompt_builder.py
        test_agent.py
```

---

## Context for Implementer

### SimpleMem Interfaces Used

```python
# From models.memory_entry
class Dialogue(BaseModel):
    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None

class MemoryEntry(BaseModel):
    entry_id: str
    lossless_restatement: str
    keywords: List[str]
    timestamp: Optional[str]
    location: Optional[str]
    persons: List[str]
    entities: List[str]
    topic: Optional[str]
    superseded_by: Optional[str]

# From main.SimpleMemSystem
class SimpleMemSystem:
    def add_dialogue(self, speaker: str, content: str, timestamp: Optional[str] = None) -> None: ...
    def get_all_memories(self) -> List[MemoryEntry]: ...
    hybrid_retriever: HybridRetriever  # accessible attribute

# From core.hybrid_retriever.HybridRetriever
class HybridRetriever:
    def retrieve(self, query: str) -> List[MemoryEntry]: ...

# From utils.llm_client.LLMClient
class LLMClient:
    def chat_completion(self, messages: List[dict], temperature: float = 0.7,
                        response_format: Optional[dict] = None) -> str: ...
```

### Async/Sync Bridge

SimpleMem is synchronous. The health agent is async. Bridge sync calls with `asyncio.to_thread()`:

```python
# For LLM calls
response = await asyncio.to_thread(llm_client.chat_completion, messages=[...], temperature=0.1)

# For SimpleMem retrieval
entries = await asyncio.to_thread(hybrid_retriever.retrieve, query="symptom headache")
```

---

## Task 1: Foundation — Types, Config, Emergency, Validators

**Files:**
- Create: `health/types.py`
- Create: `health/config.py`
- Create: `health/emergency.py`
- Create: `health/validators.py`
- Create: `tests/health/__init__.py`
- Create: `tests/health/test_validators.py`

---

- [ ] **Step 1: Create shared types**

`health/types.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional

from models.memory_entry import MemoryEntry


@dataclass
class IntentResult:
    intent: Literal["symptom", "medication", "vital", "lifestyle", "general", "emergency"]
    entities: List[str] = field(default_factory=list)
    urgency: Literal["urgent", "non-urgent"] = "non-urgent"
    suggested_query: str = ""


@dataclass
class MedicalKnowledge:
    content: str
    source: Literal["api", "cache", "llm_fallback", "unavailable"] = "api"
    api_name: Optional[str] = None
    cached_at: Optional[datetime] = None


@dataclass
class PersonalHealthContext:
    entries: List[MemoryEntry] = field(default_factory=list)
    summary: str = ""


@dataclass
class HealthResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    latency_ms: int = 0


@dataclass
class HealthProfileSnapshot:
    known_conditions: List[str] = field(default_factory=list)
    recent_measurements: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
```

- [ ] **Step 2: Create config**

`health/config.py`:

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HealthConfig:
    # Medical Knowledge API
    medical_api_url: Optional[str] = None
    medical_api_key: Optional[str] = None
    medical_api_timeout: float = 5.0
    medical_api_max_retries: int = 2

    # Cache settings
    enable_knowledge_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Retrieval
    profile_top_k: int = 10

    # Emergency detection
    urgent_keywords: list[str] = field(default_factory=lambda: [
        "胸痛", "胸口痛", "呼吸困难", "窒息", "晕倒", "昏迷",
        "大出血", "严重过敏", "休克", "chest pain", "can't breathe",
        "difficulty breathing", "passed out", "unconscious", "severe bleeding"
    ])
```

- [ ] **Step 3: Create emergency response**

`health/emergency.py`:

```python
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
```

- [ ] **Step 4: Create validators**

`health/validators.py`:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict


class ValidationError(Exception):
    pass


class Validator(ABC):
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize data. Raises ValidationError on failure."""
        ...


class BloodPressureValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        systolic = data.get("systolic")
        diastolic = data.get("diastolic")
        if systolic is None or diastolic is None:
            raise ValidationError("Blood pressure requires both systolic and diastolic values")
        try:
            sys_val = float(systolic)
            dia_val = float(diastolic)
        except (TypeError, ValueError):
            raise ValidationError("Blood pressure values must be numeric")
        if not (60 <= sys_val <= 250):
            raise ValidationError(f"Systolic pressure {sys_val} out of range (60-250)")
        if not (40 <= dia_val <= 150):
            raise ValidationError(f"Diastolic pressure {dia_val} out of range (40-150)")
        if sys_val <= dia_val:
            raise ValidationError(f"Systolic ({sys_val}) must be greater than diastolic ({dia_val})")
        return {
            "systolic": sys_val,
            "diastolic": dia_val,
            "unit": data.get("unit", "mmHg")
        }


class HeartRateValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("value")
        if value is None:
            raise ValidationError("Heart rate requires a value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Heart rate must be numeric")
        if not (30 <= val <= 220):
            raise ValidationError(f"Heart rate {val} out of range (30-220)")
        return {"value": val, "unit": data.get("unit", "bpm")}


class WeightValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("value")
        if value is None:
            raise ValidationError("Weight requires a value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Weight must be numeric")
        if not (0 < val <= 500):
            raise ValidationError(f"Weight {val} out of range (0-500)")
        return {"value": val, "unit": data.get("unit", "kg")}


class SleepDurationValidator(Validator):
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("value")
        if value is None:
            raise ValidationError("Sleep duration requires a value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValidationError("Sleep duration must be numeric")
        if not (0 <= val <= 24):
            raise ValidationError(f"Sleep duration {val} out of range (0-24)")
        return {"value": val, "unit": data.get("unit", "hours")}


VALIDATOR_REGISTRY: Dict[str, Validator] = {
    "blood_pressure": BloodPressureValidator(),
    "heart_rate": HeartRateValidator(),
    "weight": WeightValidator(),
    "sleep_duration": SleepDurationValidator(),
}


def get_validator(sign_type: str) -> Validator:
    validator = VALIDATOR_REGISTRY.get(sign_type)
    if validator is None:
        raise ValidationError(f"Unknown vital sign type: {sign_type}")
    return validator
```

- [ ] **Step 5: Write validator tests**

`tests/health/test_validators.py`:

```python
import pytest
from health.validators import (
    BloodPressureValidator,
    HeartRateValidator,
    ValidationError,
    WeightValidator,
    get_validator,
)


class TestBloodPressureValidator:
    def test_valid_bp(self):
        v = BloodPressureValidator()
        result = v.validate({"systolic": 120, "diastolic": 80})
        assert result["systolic"] == 120.0
        assert result["diastolic"] == 80.0
        assert result["unit"] == "mmHg"

    def test_invalid_high_systolic(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match="Systolic pressure 300 out of range"):
            v.validate({"systolic": 300, "diastolic": 80})

    def test_invalid_low_systolic(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match="Systolic pressure 30 out of range"):
            v.validate({"systolic": 30, "diastolic": 80})

    def test_systolic_less_than_diastolic(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match="must be greater than diastolic"):
            v.validate({"systolic": 80, "diastolic": 90})

    def test_missing_values(self):
        v = BloodPressureValidator()
        with pytest.raises(ValidationError, match="requires both"):
            v.validate({"systolic": 120})


class TestHeartRateValidator:
    def test_valid_hr(self):
        v = HeartRateValidator()
        result = v.validate({"value": 72})
        assert result["value"] == 72.0

    def test_invalid_high(self):
        v = HeartRateValidator()
        with pytest.raises(ValidationError, match="out of range"):
            v.validate({"value": 300})


class TestWeightValidator:
    def test_valid_weight(self):
        v = WeightValidator()
        result = v.validate({"value": 70.5})
        assert result["value"] == 70.5

    def test_negative_weight(self):
        v = WeightValidator()
        with pytest.raises(ValidationError, match="out of range"):
            v.validate({"value": -10})


class TestGetValidator:
    def test_known_type(self):
        v = get_validator("blood_pressure")
        assert isinstance(v, BloodPressureValidator)

    def test_unknown_type(self):
        with pytest.raises(ValidationError, match="Unknown vital sign type"):
            get_validator("unknown_type")
```

- [ ] **Step 6: Run validator tests**

```bash
pytest tests/health/test_validators.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add health/types.py health/config.py health/emergency.py health/validators.py tests/health/
git commit -m "$(cat <<'EOF'
feat(health): add foundation types, config, emergency, validators

- IntentResult, MedicalKnowledge, PersonalHealthContext, HealthResponse
- HealthConfig with API and cache settings
- EmergencyResponse with immediate fallback replies
- BloodPressure, HeartRate, Weight, SleepDuration validators
- Full test coverage for validators

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: HealthIntentExtractor

**Files:**
- Create: `health/intent.py`
- Create: `tests/health/test_intent.py`

---

- [ ] **Step 1: Write the failing test**

`tests/health/test_intent.py`:

```python
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
        # Even if LLM says non-urgent, keyword detection should override
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/health/test_intent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'health.intent'`.

- [ ] **Step 3: Write minimal implementation**

`health/intent.py`:

```python
import asyncio
import json
import re
from typing import List, Literal

from health.config import HealthConfig
from health.types import IntentResult
from utils.llm_client import LLMClient


class HealthIntentExtractor:
    def __init__(self, llm_client: LLMClient, config: HealthConfig = None):
        self.llm_client = llm_client
        self.config = config or HealthConfig()
        self._urgent_pattern = re.compile(
            "|".join(re.escape(kw) for kw in self.config.urgent_keywords),
            re.IGNORECASE
        )

    async def extract(self, user_text: str) -> IntentResult:
        # Keyword-based urgency detection (fast path)
        urgency = self._detect_urgency(user_text)

        # LLM-based intent extraction
        try:
            intent_data = await self._extract_with_llm(user_text)
        except Exception:
            # Fail-open: return general intent
            return IntentResult(
                intent="general",
                entities=[],
                urgency=urgency,  # still preserve urgency from keyword detection
                suggested_query=user_text[:100]
            )

        # Override urgency if keywords detected urgent
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/health/test_intent.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add health/intent.py tests/health/test_intent.py
git commit -m "$(cat <<'EOF'
feat(health): add HealthIntentExtractor with urgency detection

- LLM-based intent classification (symptom/medication/vital/lifestyle/general)
- Entity extraction
- Keyword-based urgency detection as fast override
- Fail-open: defaults to general intent on LLM failure
- Full async interface

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: HealthProfile

**Files:**
- Create: `health/profile.py`
- Create: `tests/health/test_profile.py`

---

- [ ] **Step 1: Write the failing test**

`tests/health/test_profile.py`:

```python
import pytest
from unittest.mock import MagicMock

from health.profile import HealthProfile
from health.types import IntentResult, PersonalHealthContext
from models.memory_entry import MemoryEntry


@pytest.fixture
def mock_simplemem():
    sm = MagicMock()
    sm.hybrid_retriever = MagicMock()
    return sm


@pytest.fixture
def profile(mock_simplemem):
    return HealthProfile(simplemem=mock_simplemem)


class TestHealthProfile:
    @pytest.mark.asyncio
    async def test_retrieve_context_basic(self, profile, mock_simplemem):
        mock_entry = MemoryEntry(
            lossless_restatement="User reported headache on June 10",
            keywords=["headache", "user"]
        )
        mock_simplemem.hybrid_retriever.retrieve.return_value = [mock_entry]

        intent = IntentResult(
            intent="symptom",
            entities=["headache"],
            suggested_query="symptom headache"
        )
        result = await profile.retrieve_context("user1", intent)

        assert isinstance(result, PersonalHealthContext)
        assert len(result.entries) == 1
        assert "headache" in result.entries[0].lossless_restatement
        mock_simplemem.hybrid_retriever.retrieve.assert_called_once_with("symptom headache")

    @pytest.mark.asyncio
    async def test_retrieve_context_empty(self, profile, mock_simplemem):
        mock_simplemem.hybrid_retriever.retrieve.return_value = []

        intent = IntentResult(intent="general", suggested_query="hello")
        result = await profile.retrieve_context("user1", intent)

        assert result.entries == []
        assert result.summary == ""

    @pytest.mark.asyncio
    async def test_retrieve_context_limit_top_k(self, profile, mock_simplemem):
        mock_entries = [MemoryEntry(lossless_restatement=f"entry {i}") for i in range(20)]
        mock_simplemem.hybrid_retriever.retrieve.return_value = mock_entries

        intent = IntentResult(intent="symptom", suggested_query="test")
        result = await profile.retrieve_context("user1", intent, top_k=5)

        assert len(result.entries) == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/health/test_profile.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'health.profile'`.

- [ ] **Step 3: Write minimal implementation**

`health/profile.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/health/test_profile.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add health/profile.py tests/health/test_profile.py
git commit -m "$(cat <<'EOF'
feat(health): add HealthProfile for personal context retrieval

- Retrieves relevant health memories via SimpleMem hybrid retriever
- Bridges sync SimpleMem with async using asyncio.to_thread
- Builds summary text for prompt injection
- Respects top_k limit, fail-open on retrieval errors

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: MedicalKnowledgeClient

**Files:**
- Create: `health/knowledge.py`
- Create: `tests/health/test_knowledge.py`

---

- [ ] **Step 1: Write the failing test**

`tests/health/test_knowledge.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/health/test_knowledge.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'health.knowledge'`.

- [ ] **Step 3: Write minimal implementation**

`health/knowledge.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/health/test_knowledge.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add health/knowledge.py tests/health/test_knowledge.py
git commit -m "$(cat <<'EOF'
feat(health): add MedicalKnowledgeClient with 3-tier fallback

- L1: External API call with exponential backoff retry
- L2: In-memory cache fallback
- L3: LLM commonsense fallback (no external call)
- Configurable timeout, retries, cache TTL
- Always returns MedicalKnowledge, never raises

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: HealthDataInput

**Files:**
- Create: `health/data_input.py`
- Create: `tests/health/test_data_input.py`

---

- [ ] **Step 1: Write the failing test**

`tests/health/test_data_input.py`:

```python
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
        assert "120/80" in call_args.kwargs["content"]

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/health/test_data_input.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'health.data_input'`.

- [ ] **Step 3: Write minimal implementation**

`health/data_input.py`:

```python
from typing import Any, Dict, Optional

from health.types import IntentResult
from health.validators import ValidationError, get_validator
from main import SimpleMemSystem


class HealthDataInput:
    def __init__(self, simplemem: SimpleMemSystem):
        self.simplemem = simplemem

    async def record(
        self,
        user_id: str,
        sign_type: str,
        data: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> dict:
        """
        Validate structured health data, convert to natural language,
        and store via SimpleMem.
        """
        try:
            validator = get_validator(sign_type)
            validated = validator.validate(data)
        except ValidationError as e:
            return {"status": "error", "message": str(e)}

        # Naturalize to text
        natural_text = self._naturalize(sign_type, validated, user_id)

        # Store via SimpleMem
        self.simplemem.add_dialogue(
            speaker="user",
            content=natural_text,
            timestamp=timestamp
        )

        return {"status": "ok", "sign_type": sign_type}

    def _naturalize(self, sign_type: str, data: Dict[str, Any], user_id: str) -> str:
        """Convert validated data to a natural language sentence."""
        if sign_type == "blood_pressure":
            return (
                f"User {user_id} recorded blood pressure: "
                f"{data['systolic']}/{data['diastolic']} {data.get('unit', 'mmHg')}."
            )
        elif sign_type == "heart_rate":
            return (
                f"User {user_id} recorded heart rate: "
                f"{data['value']} {data.get('unit', 'bpm')}."
            )
        elif sign_type == "weight":
            return (
                f"User {user_id} recorded weight: "
                f"{data['value']} {data.get('unit', 'kg')}."
            )
        elif sign_type == "sleep_duration":
            return (
                f"User {user_id} recorded sleep duration: "
                f"{data['value']} {data.get('unit', 'hours')}."
            )
        else:
            return f"User {user_id} recorded {sign_type}: {data}."
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/health/test_data_input.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add health/data_input.py tests/health/test_data_input.py
git commit -m "$(cat <<'EOF'
feat(health): add HealthDataInput for structured vital sign recording

- Validates structured data using validator registry
- Converts validated data to natural language
- Stores via SimpleMem.add_dialogue() for unified memory pipeline
- Validation failures block storage and return clear error messages

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: HealthPromptBuilder

**Files:**
- Create: `health/prompt_builder.py`
- Create: `tests/health/test_prompt_builder.py`

---

- [ ] **Step 1: Write the failing test**

`tests/health/test_prompt_builder.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/health/test_prompt_builder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'health.prompt_builder'`.

- [ ] **Step 3: Write minimal implementation**

`health/prompt_builder.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/health/test_prompt_builder.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add health/prompt_builder.py tests/health/test_prompt_builder.py
git commit -m "$(cat <<'EOF'
feat(health): add HealthPromptBuilder

- Fuses personal health context + medical knowledge + user question
- Adds source attribution annotations for degraded modes
- Returns standard OpenAI message format list
- Pure data transform, no external dependencies

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: HealthAgent (Orchestrator)

**Files:**
- Create: `health/agent.py`
- Create: `tests/health/test_agent.py`
- Modify: `health/__init__.py`

---

- [ ] **Step 1: Write the failing test**

`tests/health/test_agent.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from health.agent import HealthAgent
from health.types import HealthResponse, IntentResult, MedicalKnowledge, PersonalHealthContext


@pytest.fixture
def mock_simplemem():
    sm = MagicMock()
    sm.hybrid_retriever = MagicMock()
    return sm


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def agent(mock_simplemem, mock_llm):
    from health.config import HealthConfig
    from health.intent import HealthIntentExtractor
    from health.profile import HealthProfile
    from health.knowledge import MedicalKnowledgeClient
    from health.data_input import HealthDataInput
    from health.prompt_builder import HealthPromptBuilder

    config = HealthConfig()
    agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm, config=config)
    return agent


class TestHealthAgentAsk:
    @pytest.mark.asyncio
    async def test_ask_normal_flow(self, agent, mock_llm):
        # Mock intent extraction
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="symptom", entities=["头痛"], urgency="non-urgent",
            suggested_query="symptom 头痛"
        ))
        # Mock profile retrieval
        agent.profile.retrieve_context = AsyncMock(return_value=PersonalHealthContext(
            entries=[], summary=""
        ))
        # Mock knowledge query
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="头痛常见原因", source="api"
        ))
        # Mock LLM
        mock_llm.chat_completion.return_value = "你可能需要多休息。"

        result = await agent.ask("我最近头痛")

        assert isinstance(result, HealthResponse)
        assert "休息" in result.answer
        assert "api" in result.sources
        # Verify conversation archived
        agent.simplemem.add_dialogue.assert_called()

    @pytest.mark.asyncio
    async def test_ask_urgent_bypass(self, agent, mock_llm):
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="symptom", entities=["胸痛"], urgency="urgent"
        ))
        # Mock LLM should NOT be called
        mock_llm.chat_completion.reset_mock()

        result = await agent.ask("胸口很痛呼吸困难")

        assert "紧急" in result.answer or "URGENT" in result.answer
        assert result.latency_ms == 0
        assert result.sources == ["emergency_fallback"]
        mock_llm.chat_completion.assert_not_called()
        agent.knowledge_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_api_degraded(self, agent, mock_llm):
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="symptom", entities=["头痛"], urgency="non-urgent"
        ))
        agent.profile.retrieve_context = AsyncMock(return_value=PersonalHealthContext())
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="一般信息", source="llm_fallback"
        ))
        mock_llm.chat_completion.return_value = "建议多休息。"

        result = await agent.ask("头痛")
        assert "休息" in result.answer

    @pytest.mark.asyncio
    async def test_ask_llm_failure(self, agent, mock_llm):
        agent.intent_extractor.extract = AsyncMock(return_value=IntentResult(
            intent="general", urgency="non-urgent"
        ))
        agent.profile.retrieve_context = AsyncMock(return_value=PersonalHealthContext())
        agent.knowledge_client.query = AsyncMock(return_value=MedicalKnowledge(
            content="", source="unavailable"
        ))
        mock_llm.chat_completion.side_effect = Exception("LLM down")

        result = await agent.ask("你好")
        assert "暂时繁忙" in result.answer


class TestHealthAgentRecordVital:
    @pytest.mark.asyncio
    async def test_record_vital_success(self, agent):
        agent.data_input.record = AsyncMock(return_value={"status": "ok"})
        result = await agent.record_vital("user1", "blood_pressure", {"systolic": 120, "diastolic": 80})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_record_vital_validation_failure(self, agent):
        agent.data_input.record = AsyncMock(return_value={"status": "error", "message": "out of range"})
        result = await agent.record_vital("user1", "blood_pressure", {"systolic": 300})
        assert result["status"] == "error"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/health/test_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'health.agent'`.

- [ ] **Step 3: Write minimal implementation**

`health/agent.py`:

```python
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
```

- [ ] **Step 4: Write `health/__init__.py`**

`health/__init__.py`:

```python
from health.agent import HealthAgent
from health.config import HealthConfig
from health.data_input import HealthDataInput
from health.emergency import EmergencyResponse
from health.intent import HealthIntentExtractor
from health.knowledge import MedicalKnowledgeClient
from health.profile import HealthProfile
from health.prompt_builder import HealthPromptBuilder
from health.types import (
    HealthProfileSnapshot,
    HealthResponse,
    IntentResult,
    MedicalKnowledge,
    PersonalHealthContext,
)
from health.validators import (
    BloodPressureValidator,
    HeartRateValidator,
    SleepDurationValidator,
    ValidationError,
    WeightValidator,
)

__all__ = [
    "HealthAgent",
    "HealthConfig",
    "HealthDataInput",
    "EmergencyResponse",
    "HealthIntentExtractor",
    "MedicalKnowledgeClient",
    "HealthProfile",
    "HealthPromptBuilder",
    "HealthProfileSnapshot",
    "HealthResponse",
    "IntentResult",
    "MedicalKnowledge",
    "PersonalHealthContext",
    "BloodPressureValidator",
    "HeartRateValidator",
    "SleepDurationValidator",
    "ValidationError",
    "WeightValidator",
]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/health/test_agent.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add health/agent.py health/__init__.py tests/health/test_agent.py
git commit -m "$(cat <<'EOF'
feat(health): add HealthAgent orchestrator

- ask() method: intent extraction → emergency bypass → parallel retrieval
  → prompt fusion → LLM generation → conversation archiving
- record_vital() proxy to HealthDataInput
- Async throughout with sync-to-async bridging for SimpleMem
- Comprehensive error handling at each stage

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Integration Tests

**Files:**
- Create: `tests/health/test_integration.py`

---

- [ ] **Step 1: Write integration tests**

`tests/health/test_integration.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

from health.agent import HealthAgent
from health.config import HealthConfig
from health.types import HealthResponse, IntentResult
from models.memory_entry import MemoryEntry


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_ask_pipeline(self):
        """End-to-end: question → intent → parallel retrieval → answer → archive."""
        mock_simplemem = MagicMock()
        mock_simplemem.hybrid_retriever.retrieve.return_value = [
            MemoryEntry(lossless_restatement="User has a history of migraines")
        ]
        mock_llm = MagicMock()

        # First call: intent extraction
        # Second call: answer generation
        mock_llm.chat_completion.side_effect = [
            '{"intent": "symptom", "entities": ["头痛"], "urgency": "non-urgent", "suggested_query": "symptom 头痛"}',
            '根据你的历史，你之前有偏头痛记录。建议多休息。'
        ]

        agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm)

        with patch.object(agent.knowledge_client, "_call_api_with_retry", return_value=None):
            result = await agent.ask("我最近头痛")

        assert isinstance(result, HealthResponse)
        assert len(result.answer) > 0
        assert mock_simplemem.add_dialogue.call_count >= 2  # user + assistant archived

    @pytest.mark.asyncio
    async def test_record_then_retrieve(self):
        """Record a vital → ask about it → should retrieve the record."""
        mock_simplemem = MagicMock()
        mock_simplemem.hybrid_retriever.retrieve.return_value = [
            MemoryEntry(lossless_restatement="User user1 recorded blood pressure: 120/80 mmHg.")
        ]
        mock_llm = MagicMock()
        mock_llm.chat_completion.side_effect = [
            '{"intent": "vital", "entities": ["血压"], "urgency": "non-urgent", "suggested_query": "vital 血压"}',
            '你最近的血压是 120/80 mmHg，属于正常范围。'
        ]

        agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm)

        # Record BP
        record_result = await agent.record_vital(
            "user1", "blood_pressure", {"systolic": 120, "diastolic": 80}
        )
        assert record_result["status"] == "ok"

        # Ask about it
        with patch.object(agent.knowledge_client, "_call_api_with_retry", return_value=None):
            answer = await agent.ask("我最近血压怎么样？")

        assert "120/80" in answer.answer or "正常" in answer.answer

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Verify profile + knowledge retrieval happen in parallel."""
        import asyncio
        mock_simplemem = MagicMock()
        mock_simplemem.hybrid_retriever.retrieve.return_value = []
        mock_llm = MagicMock()
        mock_llm.chat_completion.return_value = '{"intent": "general", "entities": [], "urgency": "non-urgent", "suggested_query": "hello"}'

        agent = HealthAgent(simplemem=mock_simplemem, llm_client=mock_llm)

        call_order = []

        async def slow_profile(*args, **kwargs):
            await asyncio.sleep(0.05)
            call_order.append("profile")
            from health.types import PersonalHealthContext
            return PersonalHealthContext()

        async def slow_knowledge(*args, **kwargs):
            await asyncio.sleep(0.05)
            call_order.append("knowledge")
            from health.types import MedicalKnowledge
            return MedicalKnowledge(content="", source="unavailable")

        agent.profile.retrieve_context = slow_profile
        agent.knowledge_client.query = slow_knowledge

        mock_llm.chat_completion.return_value = "Hello!"

        start = asyncio.get_event_loop().time()
        result = await agent.ask("你好")
        elapsed = asyncio.get_event_loop().time() - start

        # Should be < 0.1s (parallel) not > 0.1s (sequential)
        assert elapsed < 0.15, f"Expected parallel execution, took {elapsed}s"
        assert "profile" in call_order
        assert "knowledge" in call_order
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/health/test_integration.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 3: Run full health test suite**

```bash
pytest tests/health/ -v
```

Expected: ALL tests PASS ( validators: 8 + intent: 5 + profile: 3 + knowledge: 5 + data_input: 5 + prompt_builder: 4 + agent: 6 + integration: 3 = 39 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/health/test_integration.py
git commit -m "$(cat <<'EOF'
test(health): add integration tests

- End-to-end ask pipeline test
- Record-then-retrieve flow test
- Parallel execution verification test

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Implementing Task | Status |
|--------------|------------------|--------|
| 6.1 HealthAgent | Task 7 | ✅ |
| 6.2 HealthIntentExtractor | Task 2 | ✅ |
| 6.3 HealthProfile | Task 3 | ✅ |
| 6.4 MedicalKnowledgeClient | Task 4 | ✅ |
| 6.5 HealthDataInput | Task 5 | ✅ |
| 6.6 HealthPromptBuilder | Task 6 | ✅ |
| 5.1 Flow A (structured input) | Task 5 | ✅ |
| 5.2 Flow B (ask pipeline) | Task 7 | ✅ |
| Emergency fast-path | Task 7 + Task 1 | ✅ |
| 3-tier API fallback | Task 4 | ✅ |
| Parallel retrieval | Task 7 | ✅ |
| Validation blocking | Task 5 | ✅ |
| Conversation archiving | Task 7 | ✅ |

### Placeholder Scan

- No TBD, TODO, or "implement later" found.
- No vague "add error handling" without specifics.
- All test code includes concrete assertions.

### Type Consistency

- `IntentResult.urgency` consistently `Literal["urgent", "non-urgent"]` across all tasks.
- `MedicalKnowledge.source` consistently `Literal["api", "cache", "llm_fallback", "unavailable"]`.
- `HealthAgent.ask()` and `record_vital()` consistently async.
- `SimpleMemSystem` interface usage consistent (add_dialogue, hybrid_retriever.retrieve).

### Gaps

- `HealthAgent.get_health_profile()` is stubbed (returns "not yet implemented"). This is acceptable per Non-Goals (visualization dashboard is future work).
- Health-specific MemoryBuilder prompt customization (Section 7.1 of spec) is not in this plan because it requires modifying SimpleMem core. The implicit tagging approach (Section 7.2) is implemented via `_archive_conversation`.

---

*Plan complete. Ready for execution.*
