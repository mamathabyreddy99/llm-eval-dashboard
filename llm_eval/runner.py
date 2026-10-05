"""Load a test suite and run every case against every model."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from llm_eval.scorers import known_scorers, score
from llm_eval.schema import Case, Result


def load_suite(path: str) -> Tuple[List[Case], str]:
    raw = Path(path).read_text(encoding="utf-8")
    cases = [Case(**item) for item in json.loads(raw)]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case ids in suite")
    unknown = {c.scorer for c in cases} - known_scorers()
    if unknown:
        raise ValueError(f"Unknown scorers in suite: {sorted(unknown)}")
    return cases, hashlib.sha256(raw.encode()).hexdigest()[:12]


def _run_one(model, case: Case, judge) -> Result:
    resp = model.generate(case)
    cost = model.cost(resp.prompt_tokens, resp.completion_tokens)
    base = dict(model=model.name, case_id=case.id, category=case.category, scorer=case.scorer,
                prompt=case.prompt, output=resp.text, latency_ms=resp.latency_ms,
                prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                cost_usd=cost, error=resp.error)
    if resp.error:
        return Result(passed=False, score=0.0, detail=f"model error: {resp.error}", **base)
    try:
        passed, value, detail = score(case, resp.text, judge)
    except Exception as exc:  # a broken scorer must not crash the whole run
        return Result(passed=False, score=0.0, detail=f"scorer error: {exc}", **base)
    return Result(passed=passed, score=value, detail=detail, **base)


def run_suite(cases: List[Case], models: list, judge: Optional[object] = None,
              workers: int = 4) -> List[Result]:
    jobs = [(m, c) for m in models for c in cases]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda job: _run_one(job[0], job[1], judge), jobs))
