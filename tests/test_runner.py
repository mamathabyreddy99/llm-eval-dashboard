from conftest import SUITE
from llm_eval.metrics import percentile, summarize
from llm_eval.models import build_mock
from llm_eval.runner import load_suite, run_suite


def run(simulate_regression=False):
    cases, _ = load_suite(SUITE)
    models = [build_mock(n, simulate_regression) for n in ("mock-small", "mock-medium", "mock-large")]
    return cases, run_suite(cases, models, workers=2)


def test_suite_loads_with_unique_ids_and_known_scorers():
    cases, suite_hash = load_suite(SUITE)
    assert len(cases) >= 30 and len(suite_hash) == 12


def test_every_model_runs_every_case():
    cases, results = run()
    assert len(results) == 3 * len(cases)


def test_runs_are_deterministic():
    _, a = run()
    _, b = run()
    assert [(r.model, r.case_id, r.passed, r.latency_ms) for r in a] == \
           [(r.model, r.case_id, r.passed, r.latency_ms) for r in b]


def test_bigger_mock_models_score_higher():
    _, results = run()
    models = summarize(results)["models"]
    assert models["mock-large"]["pass_rate"] > models["mock-small"]["pass_rate"]


def test_simulated_regression_hurts_mock_medium_only():
    _, good = run()
    _, bad = run(simulate_regression=True)
    g, b = summarize(good)["models"], summarize(bad)["models"]
    assert b["mock-medium"]["pass_rate"] < g["mock-medium"]["pass_rate"]
    assert b["mock-medium"]["p95_latency_ms"] > g["mock-medium"]["p95_latency_ms"]
    assert b["mock-large"]["pass_rate"] == g["mock-large"]["pass_rate"]


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10], 95) == 10
