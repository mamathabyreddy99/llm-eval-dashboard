"""SQLite storage so every run is kept and trends can be charted."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from llm_eval.schema import Result

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, suite TEXT NOT NULL,
    suite_hash TEXT NOT NULL, label TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
    model TEXT NOT NULL, case_id TEXT NOT NULL, category TEXT NOT NULL, scorer TEXT NOT NULL,
    prompt TEXT NOT NULL, output TEXT NOT NULL, passed INTEGER NOT NULL, score REAL NOT NULL,
    detail TEXT NOT NULL, latency_ms REAL NOT NULL, prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
"""

COLUMNS = ["model", "case_id", "category", "scorer", "prompt", "output", "passed", "score",
           "detail", "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd", "error"]


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_run(conn, run_id: str, created_at: str, suite: str, suite_hash: str,
             label: Optional[str], results: List[Result]) -> None:
    conn.execute("INSERT INTO runs (run_id, created_at, suite, suite_hash, label) VALUES (?,?,?,?,?)",
                 (run_id, created_at, suite, suite_hash, label))
    rows = [(run_id, r.model, r.case_id, r.category, r.scorer, r.prompt, r.output, int(r.passed),
             r.score, r.detail, r.latency_ms, r.prompt_tokens, r.completion_tokens, r.cost_usd,
             r.error) for r in results]
    conn.executemany(
        "INSERT INTO results (run_id, " + ", ".join(COLUMNS) + ") VALUES (?" + ",?" * len(COLUMNS) + ")",
        rows)
    conn.commit()


def latest_run(conn) -> Optional[Dict]:
    row = conn.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def load_results(conn, run_id: str) -> List[Result]:
    rows = conn.execute("SELECT * FROM results WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    out = []
    for row in rows:
        data = {c: row[c] for c in COLUMNS}
        data["passed"] = bool(data["passed"])
        out.append(Result(**data))
    return out


def history(conn, limit: int = 20) -> List[Dict]:
    rows = conn.execute(
        "SELECT r.run_id, r.created_at, x.model, AVG(x.passed) AS rate FROM runs r "
        "JOIN results x ON x.run_id = r.run_id GROUP BY r.run_id, x.model "
        "ORDER BY r.created_at, r.rowid").fetchall()
    runs: Dict[str, Dict] = {}
    for row in rows:
        entry = runs.setdefault(row["run_id"], {"run_id": row["run_id"],
                                                "created_at": row["created_at"], "rates": {}})
        entry["rates"][row["model"]] = row["rate"]
    return list(runs.values())[-limit:]
