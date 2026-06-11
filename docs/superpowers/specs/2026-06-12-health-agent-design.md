# Health Personal Assistant — Layered Domain Architecture Design

**Date:** 2026-06-12  
**Status:** Approved  
**Based on:** SimpleMem (memory layer unchanged) + 6 new application-layer modules

---

## 1. Overview

A personal health assistant agent built on top of SimpleMem. SimpleMem provides the complete lifelong memory infrastructure (semantic compression, online synthesis, hybrid retrieval, deduplication). The health assistant adds domain-specific layers for intent recognition, structured health data input, external medical knowledge queries, and personalized answer generation.

**Key principle:** SimpleMem is treated as an opaque memory layer. The health agent only interacts with `SimpleMemSystem` through its public API (`add_dialogue`, `ask`, `get_all_memories`). No modifications to SimpleMem core code.

---

## 2. Goals

1. Multi-turn health consultation with persistent memory of user health history, preferences, symptoms, medications, and measurements.
2. Structured health data input (blood pressure, weight, sleep, etc.) with validation, stored through SimpleMem's dialogue pipeline.
3. Semi-active personalization: on each user query, the agent retrieves personal health context from SimpleMem and blends it with external medical knowledge.
4. Emergency intent fast-path: urgent symptoms (chest pain, difficulty breathing) bypass LLM/API calls and return an immediate standard emergency response.
5. Fail-open external knowledge: if medical APIs are unreachable, degrade gracefully through cached knowledge or general LLM medical commonsense.

## 3. Non-Goals

- Medical diagnosis or treatment prescription (general health information only).
- Multi-user / authentication (single-user deployment for now).
- Real-time wearable integration or continuous monitoring.
- Mobile native app or complex frontend (CLI or minimal web UI out of scope for this spec).
- Physical deletion of health records (soft deletes via SimpleMem `superseded_by` only).

---

## 4. Architecture

```
User Input (question or structured data)
    │
    ▼
┌─────────────────────────────────────────────┐
│  HealthAgent (orchestrator)                 │
│  ├── ask(user_question)                     │
│  │      ├── HealthIntentExtractor           │
│  │      │      └── IntentResult             │
│  │      │           (intent, entities,      │
│  │      │            urgency, suggested_query)
│  │      ├── [if urgent] → EmergencyResponse │
│  │      ├── HealthProfile                   │
│  │      │      └── PersonalHealthContext    │
│  │      ├── MedicalKnowledgeClient          │
│  │      │      └── MedicalKnowledge         │
│  │      │           (content, source)        │
│  │      ├── HealthPromptBuilder             │
│  │      │      └── LLM prompt               │
│  │      └── LLMClient → answer              │
│  │                                           │
│  └── record_vital(sign_type, data)          │
│         └── HealthDataInput                 │
│                └── Dialogue → SimpleMem     │
└─────────────────────┬───────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                 ▼
SimpleMemSystem   External APIs    LLMClient
(Memory Layer)    (Medical KB)     (Generation)
```

### Module Boundaries

| Module | Responsibility | Dependencies |
|--------|---------------|-------------|
| `HealthAgent` | Main entry point. Orchestrates the full `ask()` and `record_vital()` flows. Handles emergency bypass. | All other modules + `SimpleMemSystem` |
| `HealthIntentExtractor` | Classifies user input into health intent, extracts entities, flags urgency, and produces an optimized retrieval query. | `LLMClient` (lightweight prompt) |
| `HealthProfile` | Retrieves and aggregates personal health context from SimpleMem using the optimized query from the intent extractor. | `SimpleMemSystem` |
| `MedicalKnowledgeClient` | Queries external medical APIs based on intent. Implements 3-tier fallback (retry → cache → LLM commonsense). | `config` (API keys), optional cache |
| `HealthDataInput` | Validates structured health data, converts to natural language, and stores as Dialogue in SimpleMem. | `SimpleMemSystem` |
| `HealthPromptBuilder` | Fuses user question + personal context + medical knowledge into the final LLM prompt. Adds source attribution annotations. | None (pure data transform) |

---

## 5. Data Flow

### 5.1 Flow A: Structured Health Data Input

```
User / Caller
    │
    ▼
HealthAgent.record_vital(
    sign_type="blood_pressure",
    data={"systolic": 120, "diastolic": 80, "unit": "mmHg"},
    timestamp="2026-06-12T09:00:00"
)
    │
    ▼
HealthDataInput.record()
    ├── [Validation] BloodPressureValidator checks range (60-250)
    │      └── FAIL (e.g. systolic=300) → return error, BLOCK storage
    │
    └── [Naturalization]
           "User recorded blood pressure: 120/80 mmHg on 2026-06-12 at 09:00."
    │
    ▼
Dialogue(speaker="user", content=natural_text, timestamp=...)
    │
    ▼
SimpleMemSystem.add_dialogue()
    → MemoryBuilder extracts → VectorStore stores
```

**Key invariant:** All structured data flows through SimpleMem's dialogue pipeline so that semantic compression and deduplication apply uniformly. A duplicate blood pressure reading on the same day will be caught by `deduplicate_entries()`.

### 5.2 Flow B: User Health Question (Semi-Active Mode)

```
User: "我最近头痛是怎么回事？"
    │
    ▼
HealthAgent.ask(user_question)
    │
    ├── Step 1: HealthIntentExtractor.extract()
    │      Input:  "我最近头痛是怎么回事？"
    │      Output: IntentResult(
    │                  intent="symptom",
    │                  entities=["头痛"],
    │                  urgency="non-urgent",
    │                  suggested_query="symptom 头痛 健康 用药 睡眠 饮食"
    │               )
    │
    ├── [URGENCY CHECK]
    │      IF urgency == "urgent":
    │          → return EmergencyResponse.standard_reply()
    │          → SKIP all subsequent steps
    │
    ├── Step 2: PARALLEL retrieval (asyncio.gather)
    │      ├─ profile_task  = HealthProfile.retrieve_context(
    │      │                   intent_result.suggested_query)
    │      │                   → PersonalHealthContext
    │      │
    │      └─ knowledge_task = MedicalKnowledgeClient.query(
    │                            intent_result)
    │                            → MedicalKnowledge
    │
    ├── Step 3: HealthPromptBuilder.build()
    │      Fuses: user_question + personal_context + medical_knowledge
    │      Produces: final LLM messages list
    │      Annotates source if medical_knowledge.source != "api"
    │
    ├── Step 4: LLMClient.chat_completion(prompt)
    │      → Generated answer
    │
    └── Step 5: Archive conversation
           Dialogue(speaker="user", content=question)
           Dialogue(speaker="assistant", content=answer)
           → SimpleMemSystem.add_dialogue()  (each)
```

**Latency target:** normal `ask()` ≈ max(memory retrieval, API call) + LLM generation. Emergency path < 10ms.

---

## 6. Core Module Interfaces

### 6.1 HealthAgent

```python
class HealthAgent:
    def __init__(self, simplemem: SimpleMemSystem, config: HealthConfig): ...

    async def ask(self, user_question: str, user_id: str = "default") -> HealthResponse:
        """
        Main entry for user questions.
        1. Extract intent (with urgency detection).
        2. If urgent → return emergency reply immediately.
        3. Otherwise → parallel retrieve personal context + medical knowledge.
        4. Build prompt → LLM generate → archive conversation.
        """

    async def record_vital(
        self, user_id: str, sign_type: str, data: dict, timestamp: Optional[str] = None
    ) -> dict:
        """
        Structured health data entry.
        Validates → naturalizes → stores via SimpleMem.
        Returns {"status": "ok"} or {"status": "error", "message": ...}.
        """

    def get_health_profile(self, user_id: str) -> HealthProfileSnapshot:
        """Aggregated health profile summary for display."""
```

### 6.2 HealthIntentExtractor

```python
@dataclass
class IntentResult:
    intent: Literal["symptom", "medication", "vital", "lifestyle", "general", "emergency"]
    entities: List[str]
    urgency: Literal["urgent", "non-urgent"]
    suggested_query: str  # Optimized for memory retrieval

class HealthIntentExtractor:
    def __init__(self, llm_client: LLMClient): ...

    async def extract(self, user_text: str) -> IntentResult:
        """
        Lightweight LLM prompt to classify intent, extract entities,
        detect urgency, and produce a retrieval-optimized query string.
        """
```

### 6.3 HealthProfile

```python
@dataclass
class PersonalHealthContext:
    entries: List[MemoryEntry]
    summary: str  # Pre-formatted text for prompt injection

class HealthProfile:
    def __init__(self, simplemem: SimpleMemSystem): ...

    async def retrieve_context(
        self, user_id: str, intent_result: IntentResult, top_k: int = 10
    ) -> PersonalHealthContext:
        """
        Uses intent_result.suggested_query to search SimpleMem.
        Returns structured personal health background.
        """
```

### 6.4 MedicalKnowledgeClient

```python
@dataclass
class MedicalKnowledge:
    content: str
    source: Literal["api", "cache", "llm_fallback", "unavailable"]
    api_name: Optional[str] = None
    cached_at: Optional[datetime] = None

class MedicalKnowledgeClient:
    def __init__(self, config: HealthConfig, cache: Optional[Cache] = None): ...

    async def query(self, intent_result: IntentResult) -> MedicalKnowledge:
        """
        3-tier fallback:
        L1: Call external API with exponential backoff (2 retries).
        L2: Return cached knowledge for this intent (if any).
        L3: Return LLM fallback with source="llm_fallback".
        Never raises. Always returns a MedicalKnowledge object.
        """
```

### 6.5 HealthDataInput

```python
class HealthDataInput:
    VALIDATORS: Dict[str, Validator] = {
        "blood_pressure": BloodPressureValidator(),
        "heart_rate": HeartRateValidator(),
        "weight": WeightValidator(),
        "sleep_duration": SleepDurationValidator(),
    }

    async def record(
        self, user_id: str, sign_type: str, data: dict, timestamp: Optional[str] = None
    ) -> dict:
        """
        1. Select validator by sign_type.
        2. Validate → FAIL returns error, does NOT store.
        3. Success → naturalize to text → Dialogue → SimpleMem.add_dialogue().
        """
```

### 6.6 HealthPromptBuilder

```python
class HealthPromptBuilder:
    SYSTEM_PROMPT: str = """You are a personal health assistant.
Constraints:
- Do NOT provide medical diagnoses or prescribe treatments.
- If symptoms are serious, strongly recommend seeing a healthcare professional.
- Prioritize referencing the user's personal health history when relevant.
"""

    def build(
        self,
        user_question: str,
        personal_context: PersonalHealthContext,
        medical_knowledge: MedicalKnowledge,
        intent: IntentResult,
    ) -> List[dict]:
        """
        Builds final LLM prompt messages.
        Adds source attribution annotations based on medical_knowledge.source:
        - "api": normal citation
        - "cache": normal citation (user not informed of cache)
        - "llm_fallback": annotate "[Medical knowledge service unavailable; based on general medical commonsense]"
        - "unavailable": annotate "[External medical query service temporarily unavailable]"
        """
```

---

## 7. Integration with SimpleMem

### 7.1 MemoryBuilder Prompt Customization

SimpleMem's `MemoryBuilder._build_extraction_prompt()` should be **overridden or extended** for health scenarios. Key additions:

- Explicitly instruct extraction of **health entities**: symptoms, medications, measurements, allergies, lifestyle habits.
- Optional `health_category` field in the extracted JSON schema for downstream filtering.
- Example entries should include health-specific restatements (blood pressure readings, symptom descriptions).

### 7.2 Dialogue Metadata (Implicit Tagging)

SimpleMem's `Dialogue` model may not support a `metadata` field. Health context is embedded implicitly via prefixed tags:

```python
health_tag = f"[HEALTH_INTENT:{intent.intent}|URGENCY:{intent.urgency}]"
enriched_content = f"{health_tag}\n{original_content}"
dialogue = Dialogue(speaker="user", content=enriched_content, timestamp=now)
simplemem.add_dialogue(dialogue.speaker, dialogue.content, dialogue.timestamp)
```

SimpleMem's extraction pipeline will naturally include these tags in the `lossless_restatement`, making them searchable for `HealthProfile` filtering without schema changes.

---

## 8. Error Handling

| Scenario | Module | Behavior | User Perception |
|----------|--------|----------|-----------------|
| External API timeout/5xx | `MedicalKnowledgeClient` | L1 retry ×2 with exponential backoff → L2 cache → L3 LLM fallback | Normal answer; L3 annotated as general commonsense |
| External API completely unavailable | `MedicalKnowledgeClient` | Returns `source="unavailable"` | Transparent notice; conversation continues |
| SimpleMem retrieval failure | `HealthProfile` | Returns empty context; answer based on medical knowledge only | Weak degradation (less personalized) |
| LLM generation failure | `HealthAgent` | Returns preset fallback message | "Service temporarily busy, please try again later" |
| Structured data validation failure | `HealthDataInput` | Blocks storage, returns specific error | Clear message: "Systolic 300 out of range (60-250)" |
| Emergency intent detected | `HealthAgent` | Immediate `EmergencyResponse.standard_reply()`, no API/LLM calls | Instant emergency notice |
| Intent extraction failure | `HealthIntentExtractor` | Defaults to `intent="general"`, `urgency="non-urgent"` | No perceptible degradation |

---

## 9. Testing Plan

### 9.1 Unit Tests (`tests/test_health_*.py`)

| Test | Input | Expected |
|------|-------|----------|
| `test_intent_symptom` | "我最近头痛" | `intent="symptom"`, `entities=["头痛"]` |
| `test_intent_medication` | "阿司匹林有什么副作用" | `intent="medication"`, `entities=["阿司匹林"]` |
| `test_intent_urgent` | "胸口很痛呼吸困难" | `urgency="urgent"` |
| `test_intent_general` | "你好" | `intent="general"` |
| `test_api_l1_retry_success` | Mock API fails twice, succeeds third time | `source="api"` |
| `test_api_l2_cache_fallback` | Mock API always fails, cache exists | `source="cache"` |
| `test_api_l3_llm_fallback` | Mock API fails + cache miss | `source="llm_fallback"` |
| `test_vital_bp_valid` | `{"systolic": 120, "diastolic": 80}` | Validation passes, stored |
| `test_vital_bp_invalid_high` | `{"systolic": 300}` | Validation error, NOT stored |
| `test_vital_bp_invalid_low` | `{"systolic": 30}` | Validation error, NOT stored |

### 9.2 Integration Tests

| Test | Verification |
|------|-------------|
| `test_ask_normal_flow` | Full `ask()` pipeline: intent → parallel retrieval → prompt → answer → archive |
| `test_ask_urgent_bypass` | Urgent intent bypasses API and LLM, returns emergency reply immediately |
| `test_ask_api_degraded` | Degraded API path produces acceptable answer quality |
| `test_record_vital_retrieval` | Record BP → ask "What was my recent BP?" → record retrieved |
| `test_parallel_latency` | `profile` + `knowledge` parallel, total < max(single) × 1.2 |

### 9.3 Boundary & Stress Tests

| Test | Scenario |
|------|----------|
| `test_empty_question` | Empty user input string |
| `test_long_question` | 2000-character symptom description |
| `test_rapid_fire` | 10 `ask()` calls within 1 second |
| `test_simplemem_large` | 100k memories in SimpleMem, verify retrieval performance |

---

## 10. Future Extensions (Not in Scope)

- Multi-user auth with tenant isolation via `tenant_id` in CrossSessionVectorStore.
- Health data visualization dashboard (weight trends, symptom timeline).
- Medication reminder scheduler (active push instead of semi-active).
- Wearable API integration (Apple Health, Fitbit) feeding into `HealthDataInput`.
- Voice input/output interface.
- LLM-as-judge for answer quality evaluation.

---

*Design approved 2026-06-12. Next step: implementation plan via `writing-plans` skill.*
