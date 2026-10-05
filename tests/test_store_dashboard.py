from conftest import SUITE
from llm_eval import store
from llm_eval.dashboard import render
from llm_eval.metrics import summarize
from llm_eval.models import build_mock
from llm_eval.regression import Finding
from llm_eval.runner import load_suite, run_suite


def make_results():
    cases, suite_hash = load_suite(SUITE)
    return run_suite(cases, [build_mock("mock-small"), build_mock("mock-large")]), suite_hash


def test_store_roundtrip_and_history(tmp_path):
    results, suite_hash = make_results()
    conn = store.connect(str(tmp_path / "t.db"))
    store.save_run(conn, "r1", "2026-01-01T00:00:00", "suite.json", suite_hash, "first", results)
    store.save_run(conn, "r2", "2026-01-02T00:00:00", "suite.json", suite_hash, None, results)
    assert store.latest_run(conn)["run_id"] == "r2"
    loaded = store.load_results(conn, "r1")
    assert len(loaded) == len(results) and loaded[0].passed == results[0].passed
    history = store.history(conn)
    assert [h["run_id"] for h in history] == ["r1", "r2"]
    assert set(history[0]["rates"]) == {"mock-small", "mock-large"}


def test_dashboard_renders_and_escapes_html():
    results, suite_hash = make_results()
    results[0].output = "<script>alert('x')</script>"
    results[0].passed = False
    run = {"run_id": "r1", "created_at": "now", "suite": "s", "suite_hash": suite_hash, "label": None}
    page = render(run, summarize(results), results,
                  [Finding("fail", "mock-small", "Pass rate fell")], [], True)
    assert "mock-small" in page and "simulated" in page and "Pass rate fell" in page
    assert "<script>alert" not in page and "&lt;script&gt;" in page
