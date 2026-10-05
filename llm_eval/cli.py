"""Command line interface:  python -m llm_eval run | report"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from llm_eval import store
from llm_eval.dashboard import render
from llm_eval.metrics import summarize
from llm_eval.models import MOCK_SPECS, OpenAICompatibleModel, build_mock
from llm_eval.regression import compare, has_failures, make_baseline
from llm_eval.runner import load_suite, run_suite


def _endpoint(spec: List[str], what: str) -> OpenAICompatibleModel:
    if len(spec) not in (3, 4):
        raise SystemExit(f"--{what} needs: NAME BASE_URL MODEL [API_KEY_ENV_VAR]")
    name, base_url, model = spec[:3]
    key = os.environ.get(spec[3]) if len(spec) == 4 else None
    return OpenAICompatibleModel(name, base_url, model, api_key=key)


def _build_models(args) -> list:
    names = args.models
    if names is None:
        names = [] if args.endpoint else list(MOCK_SPECS)
    models = []
    for name in names:
        if name not in MOCK_SPECS:
            raise SystemExit(f"Unknown mock model '{name}'. Choose from: {', '.join(MOCK_SPECS)}")
        models.append(build_mock(name, simulate_regression=args.simulate_regression))
    for spec in args.endpoint or []:
        models.append(_endpoint(spec, "endpoint"))
    return models


def _print_summary(metrics: dict) -> None:
    print(f"\n{'model':<16}{'cases':>6}{'pass %':>9}{'p50 ms':>9}{'p95 ms':>9}{'cost $':>10}")
    for name, m in metrics["models"].items():
        print(f"{name:<16}{m['n']:>6}{m['pass_rate'] * 100:>9.1f}{m['p50_latency_ms']:>9.0f}"
              f"{m['p95_latency_ms']:>9.0f}{m['cost_usd']:>10.4f}")


def _write(path: str, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def _load_baseline(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    if not Path(path).exists():
        raise SystemExit(f"Baseline file not found: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(metrics, baseline, suite_hash, args):
    if baseline is None:
        return []
    return compare(metrics, baseline, suite_hash=suite_hash, max_pass_drop=args.max_pass_drop,
                   max_latency_increase=args.max_latency_increase,
                   check_latency=not args.no_latency_check)


def cmd_run(args) -> int:
    cases, suite_hash = load_suite(args.suite)
    models = _build_models(args)
    if not models:
        raise SystemExit("No models selected.")
    judge = _endpoint(args.judge, "judge") if args.judge else None
    baseline = _load_baseline(args.baseline)

    results = run_suite(cases, models, judge=judge, workers=args.workers)
    metrics = summarize(results)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    created_at = now.isoformat(timespec="seconds")
    conn = store.connect(args.db)
    store.save_run(conn, run_id, created_at, args.suite, suite_hash, args.label, results)

    findings = _check(metrics, baseline, suite_hash, args)
    run = {"run_id": run_id, "created_at": created_at, "suite": args.suite,
           "suite_hash": suite_hash, "label": args.label}
    _write(args.report, render(run, metrics, results, findings, store.history(conn), baseline is not None))

    _print_summary(metrics)
    for f in findings:
        print(f"[{f.severity.upper()}] {f.model}: {f.message}")
    if baseline is not None and not findings:
        print("\nNo regressions vs baseline.")
    print(f"\nDashboard: {args.report}")

    if args.save_baseline:
        _write(args.save_baseline, json.dumps(make_baseline(metrics, suite_hash, created_at), indent=2))
        print(f"Baseline saved: {args.save_baseline}")
    return 1 if has_failures(findings) else 0


def cmd_report(args) -> int:
    conn = store.connect(args.db)
    run = store.latest_run(conn)
    if run is None:
        raise SystemExit("No runs in the database yet. Run: python -m llm_eval run")
    results = store.load_results(conn, run["run_id"])
    metrics = summarize(results)
    baseline = _load_baseline(args.baseline)
    findings = _check(metrics, baseline, run["suite_hash"], args)
    _write(args.out, render(run, metrics, results, findings, store.history(conn), baseline is not None))
    print(f"Dashboard written: {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_eval", description="LLM evaluation and observability")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def thresholds(p):
        p.add_argument("--baseline", help="baseline JSON to compare against")
        p.add_argument("--max-pass-drop", type=float, default=0.05)
        p.add_argument("--max-latency-increase", type=float, default=0.30)
        p.add_argument("--no-latency-check", action="store_true")

    run = sub.add_parser("run", help="run the suite and build the dashboard")
    run.add_argument("--suite", default="suites/core_suite.json")
    run.add_argument("--models", nargs="*", help=f"mock models: {', '.join(MOCK_SPECS)}")
    run.add_argument("--endpoint", nargs="+", action="append",
                     help="real model: NAME BASE_URL MODEL [API_KEY_ENV_VAR] (repeatable)")
    run.add_argument("--judge", nargs="+", help="judge model for llm_judge cases (same format)")
    run.add_argument("--simulate-regression", action="store_true",
                     help="degrade mock-medium to demonstrate the regression gate")
    run.add_argument("--db", default="data/evals.db")
    run.add_argument("--report", default="reports/dashboard.html")
    run.add_argument("--label")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--save-baseline", help="write this run's metrics as the new baseline")
    thresholds(run)
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="rebuild the dashboard from the latest stored run")
    rep.add_argument("--db", default="data/evals.db")
    rep.add_argument("--out", default="reports/dashboard.html")
    thresholds(rep)
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
