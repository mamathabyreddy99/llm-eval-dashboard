from conftest import SUITE
from llm_eval.cli import main


def args(tmp_path, *extra):
    return ["run", "--suite", SUITE, "--db", str(tmp_path / "e.db"),
            "--report", str(tmp_path / "d.html")] + list(extra)


def test_run_writes_dashboard_and_baseline(tmp_path):
    base = tmp_path / "base.json"
    assert main(args(tmp_path, "--save-baseline", str(base))) == 0
    assert base.exists() and (tmp_path / "d.html").read_text().startswith("<!doctype html>")


def test_ci_gate_passes_on_same_code_and_fails_on_regression(tmp_path):
    base = tmp_path / "base.json"
    main(args(tmp_path, "--save-baseline", str(base)))
    assert main(args(tmp_path, "--baseline", str(base))) == 0
    assert main(args(tmp_path, "--baseline", str(base), "--simulate-regression")) == 1


def test_report_command_rebuilds_dashboard(tmp_path):
    main(args(tmp_path))
    out = tmp_path / "again.html"
    assert main(["report", "--db", str(tmp_path / "e.db"), "--out", str(out)]) == 0
    assert out.exists()
