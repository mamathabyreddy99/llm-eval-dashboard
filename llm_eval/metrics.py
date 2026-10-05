"""Aggregate raw results into per-model and per-category metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from llm_eval.schema import Result


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarize(results: List[Result]) -> Dict:
    by_model: Dict[str, List[Result]] = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)

    models = {}
    for name, rows in by_model.items():
        latencies = [r.latency_ms for r in rows]
        categories: Dict[str, List[Result]] = defaultdict(list)
        for r in rows:
            categories[r.category].append(r)
        models[name] = {
            "n": len(rows),
            "passed": sum(r.passed for r in rows),
            "pass_rate": sum(r.passed for r in rows) / len(rows),
            "avg_score": sum(r.score for r in rows) / len(rows),
            "p50_latency_ms": round(percentile(latencies, 50), 2),
            "p95_latency_ms": round(percentile(latencies, 95), 2),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            "cost_usd": sum(r.cost_usd for r in rows),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in rows),
            "error_rate": sum(1 for r in rows if r.error) / len(rows),
            "categories": {
                c: {"n": len(rs), "pass_rate": sum(r.passed for r in rs) / len(rs)}
                for c, rs in sorted(categories.items())
            },
        }
    return {"models": models}
