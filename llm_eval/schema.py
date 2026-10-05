"""Plain data classes shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Case:
    id: str
    category: str
    prompt: str
    reference: str
    scorer: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


@dataclass
class Result:
    model: str
    case_id: str
    category: str
    scorer: str
    prompt: str
    output: str
    passed: bool
    score: float
    detail: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: Optional[str] = None
