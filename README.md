# LLM Evaluation and Observability Dashboard

A small, dependency-light framework that **tests LLMs like software**: run a suite of test cases against
one or more models, score every answer, store each run, chart quality / latency / cost, and **fail the
build in CI when a model update makes things worse**.

```
suite (JSON cases) -> runner -> scorers -> SQLite history -> HTML dashboard
                          |                                      ^
                  mock or real models                 baseline regression gate (CI)
```

## Why this project
Evaluation and regression testing for LLM systems is something many teams struggle with. This project shows
you can measure model quality, latency and cost, and catch regressions automatically.

## Quick start (Mac / Linux)

```bash
cd llm-eval-dashboard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 -m pytest -q                                   # 27 tests
python3 -m llm_eval run --save-baseline baselines/baseline.json
open reports/dashboard.html                            # Windows: start reports\dashboard.html
```

Windows: use `python` instead of `python3`, and `.venv\Scripts\activate` to activate.

## Demo the regression gate

```bash
python3 -m llm_eval run --baseline baselines/baseline.json                      # exit code 0
python3 -m llm_eval run --baseline baselines/baseline.json --simulate-regression # exit code 1
```
The second command simulates a bad model update. It prints `[FAIL]` lines (pass rate, category and p95
latency regressions), shows them on the dashboard, and exits with code 1, which is how CI blocks a bad change.

## IMPORTANT: mock models vs real models
By default the run uses **mock models** (`mock-small`, `mock-medium`, `mock-large`). They are deterministic
simulations so the whole pipeline works offline and in CI. **Their accuracy and latency numbers are simulated.
Do not present them as real LLM performance.** For real numbers, point the tool at a real model:

**Free, on your laptop (Ollama)**
```bash
brew install ollama            # or download from ollama.com
ollama pull llama3.2:1b
python3 -m llm_eval run --endpoint ollama http://localhost:11434/v1 llama3.2:1b
```

**Any OpenAI-compatible API (for example Together AI)**
```bash
export TOGETHER_API_KEY=your_key
python3 -m llm_eval run --endpoint together https://api.together.xyz/v1 <model-name> TOGETHER_API_KEY
```
(Check the provider's docs for the current base URL and model names. The real-endpoint code path is
unit-tested only for its scoring side; try it on your machine and fix anything provider-specific.)

You can compare several at once by repeating `--endpoint`, or mix them with `--models mock-small`.

## What it measures
- **Quality:** pass rate overall and per category (factual, math, reasoning, format, instructions, safety, summarization)
- **Latency:** p50, p95, mean per model (real wall-clock time for real endpoints)
- **Cost:** tokens and estimated USD per model
- **Trend:** pass rate across stored runs
- **Failures:** every failing case with the output and why it failed

## Scorers
`exact_match`, `contains`, `numeric`, `json_schema`, `regex`, `refusal` (checks both refusing unsafe
prompts and *not* over-refusing safe ones), `max_words`, `similarity` (token F1), and `llm_judge`
(uses a judge model via `--judge`, falls back to similarity if none).

## Add your own tests
Append a case to `suites/core_suite.json`:
```json
{"id": "fact-07", "category": "factual", "prompt": "What is the capital of Japan?",
 "reference": "Tokyo", "scorer": "contains"}
```
Then re-run and save a new baseline (`--save-baseline`). To add a scorer, write a function in
`llm_eval/scorers.py` and register it in `SCORERS`.

## CI gate
`.github/workflows/ci.yml` runs the tests, then `python -m llm_eval run --baseline baselines/baseline.json`.
The build fails if any model's pass rate drops more than 5 points, any category drops more than 25 points,
or p95 latency rises more than 30% (tune with `--max-pass-drop`, `--max-latency-increase`,
`--no-latency-check`). Real models have noisy latency, so consider `--no-latency-check` for them.

## Project layout
```
llm_eval/   cli.py, runner.py, scorers.py, models.py, metrics.py, regression.py, store.py, dashboard.py
suites/     core_suite.json (36 cases, 7 categories)
baselines/  baseline.json
tests/      27 pytest tests
docs/       sample_dashboard.html (open it in a browser to preview)
```

## Results (fill in from YOUR real-model run)
| model | cases | pass % | p50 ms | p95 ms | cost |
|---|---|---|---|---|---|
| | | | | | |

## Design decisions and tradeoffs
- **Mock models + real adapters:** mocks keep CI fast, free and deterministic; the adapter interface makes real models a drop-in.
- **SQLite:** zero setup and enough for one machine; Postgres would be the step up for a team.
- **Static HTML dashboard:** no server or JavaScript, so it can be uploaded as a CI artifact.
- **Rule-based scorers first:** cheap, reproducible and debuggable. LLM-as-judge is available but can be biased and noisy.
- **Small suite:** 36 cases catch big regressions but give wide error bars per category; grow the suite for tighter signal.

## Ideas to extend
1. Validate the judge: label 30 answers by hand and report judge-vs-human agreement.
2. Add confidence intervals and a significance test (for example McNemar) to the regression gate.
3. Run cases repeatedly at temperature > 0 and report variance.
4. Add a prompt-versioning table to compare prompt changes, not just model changes.
5. Serve the dashboard with FastAPI and add a live run button.
