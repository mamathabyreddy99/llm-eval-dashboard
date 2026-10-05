"""Model adapters.

MockModel      - deterministic simulated models. They let the whole pipeline (scoring, storage,
                 dashboard, regression gate, CI) run offline. Their numbers are SIMULATED.
OpenAICompatibleModel - real models behind any OpenAI-style /chat/completions endpoint
                 (Ollama, vLLM, Together AI, OpenAI, ...). Use this for real measurements.
"""
from __future__ import annotations

import hashlib
import random
import time
from typing import Dict, Optional

from llm_eval.schema import Case, ModelResponse

REFUSAL = "I can't help with that request."
COMPLIANCE = "Sure, here is how you can do that: first, gather the materials, then follow these steps."
UNSURE = "I'm not sure about that."
FILLER = (
    "Well, to give a thorough answer, it is worth noting that there are many details and "
    "considerations involved here, so let me elaborate at some length on the background."
)


def _tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class BaseModel:
    name = "base"
    price_in = 0.0   # USD per 1K prompt tokens
    price_out = 0.0  # USD per 1K completion tokens

    def generate(self, case: Case) -> ModelResponse:
        raise NotImplementedError

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens / 1000 * self.price_in + completion_tokens / 1000 * self.price_out


MOCK_SPECS: Dict[str, dict] = {
    "mock-small": dict(quality=0.62, base_latency_ms=180, per_token_ms=12, price_in=0.0002, price_out=0.0004),
    "mock-medium": dict(quality=0.80, base_latency_ms=320, per_token_ms=20, price_in=0.001, price_out=0.002),
    "mock-large": dict(quality=0.93, base_latency_ms=650, per_token_ms=35, price_in=0.005, price_out=0.015),
}
CATEGORY_ADJUST = {"math": -0.10, "reasoning": -0.12}


class MockModel(BaseModel):
    def __init__(self, name, quality, base_latency_ms, per_token_ms, price_in, price_out,
                 latency_factor=1.0):
        self.name = name
        self.quality = quality
        self.base_latency_ms = base_latency_ms
        self.per_token_ms = per_token_ms
        self.price_in = price_in
        self.price_out = price_out
        self.latency_factor = latency_factor

    def _rng(self, case: Case, salt: str) -> random.Random:
        digest = hashlib.sha256(f"{self.name}|{case.id}|{salt}".encode()).hexdigest()
        return random.Random(int(digest[:12], 16))

    def _quality(self, case: Case) -> float:
        return min(1.0, max(0.0, self.quality + CATEGORY_ADJUST.get(case.category, 0.0)))

    def _good(self, case: Case) -> str:
        if case.scorer == "contains":
            return f"The answer is {case.reference}."
        if case.scorer == "numeric":
            return f"The result is {case.reference}."
        return case.reference

    def _bad(self, case: Case, rng: random.Random) -> str:
        if case.scorer == "numeric":
            try:
                value = float(case.reference) + rng.randint(1, 9)
                return f"The result is {value:g}."
            except ValueError:
                return UNSURE
        if case.scorer == "json_schema":
            return case.reference[:-2]
        if case.scorer == "refusal":
            return COMPLIANCE if case.params.get("expect_refusal", True) else REFUSAL
        if case.scorer == "max_words":
            return f"{case.reference} {FILLER}"
        return UNSURE

    def generate(self, case: Case) -> ModelResponse:
        correct = self._rng(case, "q").random() < self._quality(case)
        text = self._good(case) if correct else self._bad(case, self._rng(case, "b"))
        prompt_tokens, completion_tokens = _tokens(case.prompt), _tokens(text)
        jitter = 0.8 + 0.4 * self._rng(case, "lat").random()
        latency = (self.base_latency_ms * jitter + completion_tokens * self.per_token_ms) * self.latency_factor
        return ModelResponse(text, round(latency, 2), prompt_tokens, completion_tokens)


def build_mock(name: str, simulate_regression: bool = False) -> MockModel:
    spec = dict(MOCK_SPECS[name])
    if simulate_regression and name == "mock-medium":
        spec["quality"] -= 0.25       # the "bad model update"
        spec["latency_factor"] = 1.6
    return MockModel(name, **spec)


class OpenAICompatibleModel(BaseModel):
    def __init__(self, name, base_url, model, api_key: Optional[str] = None,
                 price_in=0.0, price_out=0.0, timeout=60.0, max_tokens=256):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.price_in = price_in
        self.price_out = price_out
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> ModelResponse:
        import httpx  # imported lazily so mock-only runs need no network library

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        start = time.perf_counter()
        try:
            resp = httpx.post(f"{self.base_url}/chat/completions", json=body,
                              headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            usage = data.get("usage") or {}
            latency = (time.perf_counter() - start) * 1000
            return ModelResponse(
                text, round(latency, 2),
                usage.get("prompt_tokens") or _tokens(prompt),
                usage.get("completion_tokens") or _tokens(text),
            )
        except Exception as exc:  # network, HTTP, or parsing problems become failed cases
            latency = (time.perf_counter() - start) * 1000
            return ModelResponse("", round(latency, 2), _tokens(prompt), 0,
                                 error=f"{type(exc).__name__}: {exc}")

    def generate(self, case: Case) -> ModelResponse:
        return self.complete(case.prompt)
