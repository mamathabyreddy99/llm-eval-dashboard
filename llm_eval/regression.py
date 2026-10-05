"""Compare a run with a saved baseline and flag regressions (used as a CI gate)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Finding:
    severity: str  # "fail" or "warn"
    model: str
    message: str


def make_baseline(metrics: Dict, suite_hash: str, created_at: str) -> Dict:
    models = {}
    for name, m in metrics["models"].items():
        models[name] = {
            "pass_rate": m["pass_rate"],
            "p95_latency_ms": m["p95_latency_ms"],
            "categories": {c: v["pass_rate"] for c, v in m["categories"].items()},
        }
    return {"suite_hash": suite_hash, "created_at": created_at, "models": models}


def compare(metrics: Dict, baseline: Dict, suite_hash: Optional[str] = None,
            max_pass_drop: float = 0.05, max_category_drop: float = 0.25,
            max_latency_increase: float = 0.30, check_latency: bool = True) -> List[Finding]:
    findings: List[Finding] = []
    if suite_hash and baseline.get("suite_hash") and suite_hash != baseline["suite_hash"]:
        findings.append(Finding("warn", "-", "Test suite changed since the baseline was saved; "
                                              "consider re-saving the baseline."))
    base_models = baseline.get("models", {})
    for name, cur in metrics["models"].items():
        base = base_models.get(name)
        if base is None:
            findings.append(Finding("warn", name, "No baseline for this model."))
            continue
        drop = base["pass_rate"] - cur["pass_rate"]
        if drop > max_pass_drop + 1e-9:
            findings.append(Finding(
                "fail", name,
                f"Pass rate fell {base['pass_rate']*100:.1f}% -> {cur['pass_rate']*100:.1f}% "
                f"(allowed drop {max_pass_drop*100:.0f} points)."))
        for cat, base_rate in base.get("categories", {}).items():
            cur_cat = cur["categories"].get(cat)
            if cur_cat and base_rate - cur_cat["pass_rate"] > max_category_drop + 1e-9:
                findings.append(Finding(
                    "fail", name,
                    f"Category '{cat}' fell {base_rate*100:.0f}% -> {cur_cat['pass_rate']*100:.0f}%."))
        base_p95 = base.get("p95_latency_ms", 0)
        if check_latency and base_p95 > 0 and cur["p95_latency_ms"] > base_p95 * (1 + max_latency_increase):
            findings.append(Finding(
                "fail", name,
                f"p95 latency rose {base_p95:.0f} ms -> {cur['p95_latency_ms']:.0f} ms "
                f"(allowed +{max_latency_increase*100:.0f}%)."))
    for name in base_models:
        if name not in metrics["models"]:
            findings.append(Finding("warn", name, "In the baseline but not in this run."))
    return findings


def has_failures(findings: List[Finding]) -> bool:
    return any(f.severity == "fail" for f in findings)
