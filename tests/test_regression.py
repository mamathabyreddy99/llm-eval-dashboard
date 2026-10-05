from llm_eval.regression import compare, has_failures, make_baseline


def metrics(rate, p95, cat_rate=None):
    cat_rate = rate if cat_rate is None else cat_rate
    return {"models": {"m": {"pass_rate": rate, "p95_latency_ms": p95,
                             "categories": {"math": {"n": 6, "pass_rate": cat_rate}}}}}


BASE = make_baseline(metrics(0.80, 500), "abc", "now")


def test_identical_run_has_no_findings():
    assert compare(metrics(0.80, 500), BASE, suite_hash="abc") == []


def test_small_drop_within_tolerance_passes():
    assert not has_failures(compare(metrics(0.77, 500), BASE))


def test_large_pass_rate_drop_fails():
    findings = compare(metrics(0.60, 500), BASE)
    assert has_failures(findings) and "Pass rate fell" in findings[0].message


def test_category_collapse_fails():
    assert has_failures(compare(metrics(0.80, 500, cat_rate=0.30), BASE))


def test_latency_spike_fails_unless_disabled():
    assert has_failures(compare(metrics(0.80, 900), BASE))
    assert not has_failures(compare(metrics(0.80, 900), BASE, check_latency=False))


def test_suite_change_and_new_model_only_warn():
    findings = compare(metrics(0.80, 500), BASE, suite_hash="different")
    assert findings and not has_failures(findings)
    other = {"models": {"new": metrics(0.5, 100)["models"]["m"]}}
    assert not has_failures(compare(other, BASE))
