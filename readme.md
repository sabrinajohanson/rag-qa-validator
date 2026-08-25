# RAG QA Validator

Validates whether LLM-generated answers stay **faithful** to the source documentation they were retrieved from, using [RAGAS](https://github.com/explodinggradients/ragas) metrics against a hand-built QA dataset with intentionally injected failure cases.

[![CI](https://github.com/sabrinajohanson/rag-qa-validator/actions/workflows/mock-ci.yml/badge.svg)](https://github.com/sabrinajohanson/rag-qa-validator/actions/workflows/mock-ci.yml)

## Why It Matters

Retrieval-Augmented Generation (RAG) systems fail in a specific and dangerous way: the retrieved context can be correct while the generated answer still drifts from it — adding facts that were never in the source, contradicting it outright, or answering a different question than the one asked. Traditional QA (assert response == expected) cannot catch this, because there is rarely a single "correct" string; the same faithful answer can be phrased a dozen different ways.

This project builds a small, structured evaluation harness around that specific gap: given a question, a retrieved context, and a generated answer, does the answer actually follow from the context? It is one of the most frequently cited requirements in current AI QA job postings, and one of the least covered by traditional QA portfolios.

## How It Works

1. **`data/qa_dataset.json`** holds 6 QA cases built around a fictional "TaskFlow API v2.1" documentation set. Each case has a question, the context that would have been retrieved, a generated answer, a ground truth, and an `expected_verdict`.
2. **`src/validator.py`** loads the dataset, sends it through RAGAS's `evaluate()`, and gets back four scores per case: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
3. **`src/metrics.py`** applies fixed pass/fail thresholds to each score and rolls them up into an overall per-case verdict.
4. **`src/main.py`** is the CLI that ties it together and prints a summary.
5. Everything is exported to a standardized **`results.json`**, matching the pattern used in `llm-test-case-generator` and `llm-as-judge`.

## Dataset Design

| ID | Targets | Expected | What's being tested |
|----|---------|----------|----------------------|
| 001 | faithfulness | pass | Clean, accurate baseline answer |
| 002 | faithfulness | fail | Hallucinated numbers not present in context (rate limit) |
| 003 | faithfulness | fail | Fabricated retry policy that contradicts the context |
| 004 | answer_relevancy | fail | Context is correct, but the answer addresses a different question |
| 005 | context_recall | fail | Retrieved context is incomplete vs. the ground truth (missing a status code) |
| 006 | faithfulness | pass | Faithful paraphrase, not verbatim — the case that triggered the threshold recalibration below |

## Metrics & Thresholds

| Metric | Threshold | What it measures |
|--------|-----------|-------------------|
| `faithfulness` | 0.80 | Does the answer avoid claims that aren't supported by the retrieved context? |
| `answer_relevancy` | 0.75 | Does the answer actually address the question asked? |
| `context_precision` | 0.70 | Is the retrieved context relevant to the question (low noise)? |
| `context_recall` | 0.70 | Does the retrieved context cover everything the ground truth needs? |

A case's overall verdict is `fail` if **any** metric falls below its threshold.

## Running Locally

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# mock tests - no API cost
python -m pytest tests/unit/ -v

# live run against real TaskFlow QA dataset - costs OpenAI API calls
set OPENAI_API_KEY=sk-...     # Windows (cmd)
# export OPENAI_API_KEY=sk-...  # macOS/Linux
python -m src.main --dataset data/qa_dataset.json --output results.json
```

## CI/CD Strategy

Same two-tier pattern used across the other AI repos in this portfolio:

- **`mock-ci.yml`** — runs on every push/PR, no API cost, exercises the threshold/verdict logic with mocked RAGAS scores, publishes the Allure report to GitHub Pages via `gh-pages`.
- **`live-ci.yml`** — `workflow_dispatch` only (manual trigger), makes real OpenAI calls through RAGAS, uploads `results.json` + Allure results as workflow artifacts. It does **not** publish to Pages, to avoid a `gh-pages` deploy collision with `mock-ci.yml`.

| Piece | GitHub-Actions-specific | Transferable to TeamCity |
|-------|--------------------------|----------------------------|
| `workflow_dispatch` trigger syntax | Yes | No — TeamCity uses a manual build configuration / trigger instead |
| `actions/deploy-pages` + `gh-pages` publish | Yes | No — TeamCity has no built-in Pages equivalent; would need a separate artifact/static host step |
| Running `pytest` + generating Allure results | No | **Yes** — identical commands, TeamCity just needs an Allure report tab/plugin |
| "Manual trigger for the cost-incurring suite, cron/push for the free suite" | No | **Yes** — this split is the actual reusable idea, independent of which CI tool renders it |

## Findings

The `faithfulness` threshold was originally set to 0.90. Case 006 (a faithful but non-verbatim paraphrase) initially scored below that line — a false negative caused by RAGAS's internal faithfulness scorer being sensitive to paraphrasing, not because the answer was actually unfaithful. The threshold was recalibrated to 0.80 after manually reviewing the case and confirming the answer introduced no new claims. This is documented here rather than silently adjusted, because a threshold change without a paper trail is exactly the kind of thing a QA report should never hide.

See also: [`llm-as-judge`](https://github.com/sabrinajohanson/llm-as-judge), where a related finding is documented — the judge model catching real classification errors in another project's output.

## Limitations

- The dataset has only 6 cases. It is designed to exercise each metric's failure mode clearly, not to provide statistical coverage of a production RAG system.
- RAGAS metrics are themselves LLM-as-judge under the hood, so scores inherit some non-determinism — a case sitting close to a threshold may occasionally flip verdicts between runs.
- The version of `ragas` pinned in `requirements.txt` unconditionally imports `ChatVertexAI` from `langchain_community.chat_models.vertexai` at import time, even for projects that never use Google VertexAI. Recent `langchain-community` releases removed that submodule, which breaks importing `ragas` entirely for OpenAI-only projects like this one ([upstream issue](https://github.com/explodinggradients/ragas/issues/2753)). `src/__init__.py` registers a lightweight stand-in module before `ragas` is imported so the import succeeds; `ChatVertexAI` is never actually instantiated anywhere in this project. This is an environment-level workaround, not a design choice — it should be removed once `ragas` ships a fix upstream.
- [DeepEval](https://github.com/confident-ai/deepeval) was considered as an alternative (it integrates natively with `pytest`), but RAGAS was chosen for this project because of its RAG-specific metric set. DeepEval remains a natural next step for broader LLM output validation beyond RAG.

## License

MIT — see [LICENSE](./LICENSE).