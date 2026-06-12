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
