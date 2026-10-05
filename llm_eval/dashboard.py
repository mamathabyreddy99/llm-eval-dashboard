"""Render a self-contained HTML dashboard (no JavaScript, no external assets)."""
from __future__ import annotations

import html
from typing import Dict, List

from llm_eval.regression import Finding
from llm_eval.schema import Result

PALETTE = ["#4f8ef7", "#f59e0b", "#10b981", "#ef4444", "#a855f7", "#14b8a6"]


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _hue(rate: float) -> int:
    return int(max(0.0, min(1.0, rate)) * 120)


def _snippet(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "..."


def _badge(name: str) -> str:
    return ' <span class="badge sim">simulated</span>' if name.startswith("mock-") else ""


def _kpis(models: Dict) -> str:
    cards = []
    for name, m in models.items():
        cards.append(
            f'<div class="card"><div class="muted">{_e(name)}{_badge(name)}</div>'
            f'<div class="big" style="color:hsl({_hue(m["pass_rate"])},60%,42%)">{_pct(m["pass_rate"])}</div>'
            f'<div class="muted">{m["passed"]}/{m["n"]} passed</div>'
            f'<div class="row"><span>p95 {m["p95_latency_ms"]:.0f} ms</span>'
            f'<span>${m["cost_usd"]:.4f}</span></div></div>')
    return '<div class="cards">' + "".join(cards) + "</div>"


def _bar_chart(models: Dict) -> str:
    rows, y = [], 10
    for i, (name, m) in enumerate(models.items()):
        w = 420 * m["pass_rate"]
        color = PALETTE[i % len(PALETTE)]
        rows.append(
            f'<text x="0" y="{y + 15}" class="svgt">{_e(name)}</text>'
            f'<rect x="130" y="{y}" width="420" height="20" rx="4" class="track"/>'
            f'<rect x="130" y="{y}" width="{w:.1f}" height="20" rx="4" fill="{color}"/>'
            f'<text x="{130 + 428}" y="{y + 15}" class="svgt">{_pct(m["pass_rate"])}</text>')
        y += 32
    return f'<svg viewBox="0 0 680 {y + 4}" class="chart" role="img" aria-label="Pass rate by model">{"".join(rows)}</svg>'


def _heatmap(models: Dict) -> str:
    names = list(models)
    cats = sorted({c for m in models.values() for c in m["categories"]})
    head = "".join(f"<th>{_e(n)}</th>" for n in names)
    body = []
    for cat in cats:
        cells = []
        for n in names:
            c = models[n]["categories"].get(cat)
            if c is None:
                cells.append("<td>-</td>")
            else:
                cells.append(f'<td style="background:hsla({_hue(c["pass_rate"])},65%,45%,0.28)">'
                             f'{c["pass_rate"] * 100:.0f}% <span class="muted">({c["n"]})</span></td>')
        body.append(f"<tr><th>{_e(cat)}</th>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><tr><th>Category</th>{head}</tr>{"".join(body)}</table></div>'


def _latency_table(models: Dict) -> str:
    top = max([m["p95_latency_ms"] for m in models.values()] + [1.0])
    rows = []
    for name, m in models.items():
        w = 100 * m["p95_latency_ms"] / top
        rows.append(
            f'<tr><th>{_e(name)}</th><td>{m["p50_latency_ms"]:.0f} ms</td>'
            f'<td>{m["p95_latency_ms"]:.0f} ms</td><td>{m["mean_latency_ms"]:.0f} ms</td>'
            f'<td><div class="mini"><div style="width:{w:.0f}%"></div></div></td>'
            f'<td>{m["total_tokens"]}</td><td>${m["cost_usd"]:.4f}</td>'
            f'<td>{_pct(m["error_rate"])}</td></tr>')
    return ('<div class="scroll"><table><tr><th>Model</th><th>p50</th><th>p95</th><th>Mean</th>'
            '<th>p95 vs slowest</th><th>Tokens</th><th>Cost</th><th>Errors</th></tr>'
            + "".join(rows) + "</table></div>")


def _trend(history: List[Dict], model_names: List[str]) -> str:
    if len(history) < 2:
        return '<p class="muted">Run the evaluation again to see pass-rate trends across runs.</p>'
    w, h, pad = 640, 220, 34
    n = len(history)
    parts = []
    for tick in (0, 0.5, 1.0):
        y = h - pad - tick * (h - 2 * pad)
        parts.append(f'<line x1="{pad}" x2="{w - 10}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="0" y="{y + 4:.1f}" class="svgt">{int(tick * 100)}%</text>')
    for i, name in enumerate(model_names):
        pts = []
        for j, run in enumerate(history):
            if name in run["rates"]:
                x = pad + j * (w - pad - 10) / (n - 1)
                y = h - pad - run["rates"][name] * (h - 2 * pad)
                pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            color = PALETTE[i % len(PALETTE)]
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>')
            parts.append(f'<text x="{pad + i * 150}" y="{h - 6}" class="svgt" style="fill:{color}">&#9632; {_e(name)}</text>')
    return f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Pass rate trend">{"".join(parts)}</svg>'


def _findings(findings: List[Finding], has_baseline: bool) -> str:
    if not has_baseline:
        return '<p class="muted">No baseline supplied. Use <code>--baseline</code> to enable regression checks.</p>'
    if not findings:
        return '<p class="ok">No regressions vs baseline.</p>'
    items = "".join(
        f'<li><span class="badge {_e(f.severity)}">{_e(f.severity.upper())}</span> '
        f'<strong>{_e(f.model)}</strong> {_e(f.message)}</li>' for f in findings)
    return f'<ul class="findings">{items}</ul>'


def _failures(results: List[Result], limit: int = 60) -> str:
    failed = [r for r in results if not r.passed]
    if not failed:
        return '<p class="ok">Every case passed.</p>'
    rows = "".join(
        f'<tr><td>{_e(r.model)}</td><td>{_e(r.case_id)}</td><td>{_e(r.category)}</td>'
        f'<td>{_e(_snippet(r.prompt, 80))}</td><td>{_e(_snippet(r.output))}</td>'
        f'<td>{_e(r.detail)}</td></tr>' for r in failed[:limit])
    note = f'<p class="muted">Showing {min(limit, len(failed))} of {len(failed)} failures.</p>' if len(failed) > limit else ""
    return (f'<div class="scroll"><table><tr><th>Model</th><th>Case</th><th>Category</th><th>Prompt</th>'
            f'<th>Output</th><th>Why it failed</th></tr>{rows}</table></div>{note}')


CSS = """
:root{--bg:#f7f8fa;--fg:#1c2330;--muted:#6b7385;--card:#fff;--line:#e3e6ec;--track:#e9ecf2}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--fg:#e6e9f0;--muted:#97a0b3;--card:#1b2029;--line:#2a3140;--track:#2a3140}}
*{box-sizing:border-box}body{margin:0;padding:24px;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1060px;margin:0 auto}h1{margin:0 0 4px;font-size:24px}h2{margin:28px 0 10px;font-size:16px}
.muted{color:var(--muted)}.ok{color:#10b981;font-weight:600}code{background:var(--track);padding:1px 5px;border-radius:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.big{font-size:32px;font-weight:700;line-height:1.2}.row{display:flex;justify-content:space-between;margin-top:8px;color:var(--muted);font-size:12px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600}
.chart{width:100%;height:auto}.svgt{font-size:12px;fill:var(--fg)}.track{fill:var(--track)}.grid{stroke:var(--line)}
.mini{background:var(--track);border-radius:4px;height:8px;min-width:90px}.mini div{background:#4f8ef7;height:8px;border-radius:4px}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;font-weight:600;background:var(--track);color:var(--muted)}
.badge.sim{background:#f59e0b22;color:#b7791f}.badge.fail{background:#ef444422;color:#dc2626}.badge.warn{background:#f59e0b22;color:#b7791f}
.findings{padding-left:18px}.findings li{margin:6px 0}
"""


def render(run: Dict, metrics: Dict, results: List[Result], findings: List[Finding],
           history: List[Dict], has_baseline: bool) -> str:
    models = metrics["models"]
    names = list(models)
    sim_note = ""
    if any(n.startswith("mock-") for n in names):
        sim_note = ('<p class="muted">Models marked <span class="badge sim">simulated</span> are deterministic '
                    'mock models used to exercise the pipeline. Run against real models for real numbers.</p>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Evaluation Dashboard</title><style>{CSS}</style></head><body><main>
<h1>LLM Evaluation Dashboard</h1>
<div class="muted">Run {_e(run["run_id"])} &middot; {_e(run["created_at"])} &middot; suite {_e(run["suite"])}
 ({_e(run["suite_hash"])}){(" &middot; " + _e(run["label"])) if run.get("label") else ""}</div>
{sim_note}
<h2>Overview</h2>{_kpis(models)}
<h2>Pass rate by model</h2><div class="panel">{_bar_chart(models)}</div>
<h2>Regression check vs baseline</h2><div class="panel">{_findings(findings, has_baseline)}</div>
<h2>Pass rate by category</h2><div class="panel">{_heatmap(models)}</div>
<h2>Latency, tokens, and cost</h2><div class="panel">{_latency_table(models)}</div>
<h2>Pass-rate trend across runs</h2><div class="panel">{_trend(history, names)}</div>
<h2>Failing cases</h2><div class="panel">{_failures(results)}</div>
</main></body></html>"""
