"""
Mock tests for the RAG QA validator - no OpenAI calls, no API cost.

Patches ragas.evaluate() and src.validator.get_judge_llm() with canned
values so the threshold/verdict logic and the results.json export shape
can be exercised on every push, without depending on a live LLM call or a
real OPENAI_API_KEY. This is the suite that runs in mock-ci.yml.
"""

import json
from unittest.mock import patch, MagicMock

import pandas as pd

from src.metrics import apply_verdict, THRESHOLDS
from src import validator


def test_apply_verdict_all_pass():
    scores = {
        "faithfulness": 0.95,
        "answer_relevancy": 0.90,
        "context_precision": 0.85,
        "context_recall": 0.80,
    }
    result = apply_verdict(scores)
    assert result["overall"] == "pass"
    assert all(m["verdict"] == "pass" for m in result["metrics"].values())


def test_apply_verdict_faithfulness_fail():
    scores = {
        "faithfulness": 0.50,  # below 0.80 threshold
        "answer_relevancy": 0.90,
        "context_precision": 0.85,
        "context_recall": 0.80,
    }
    result = apply_verdict(scores)
    assert result["overall"] == "fail"
    assert result["metrics"]["faithfulness"]["verdict"] == "fail"


def test_apply_verdict_boundary_score_passes():
    # A score exactly equal to the threshold should pass (>=, not >)
    scores = {"faithfulness": THRESHOLDS["faithfulness"]}
    result = apply_verdict(scores)
    assert result["metrics"]["faithfulness"]["verdict"] == "pass"


def test_apply_verdict_missing_metric_is_skipped():
    # If a metric key is absent from scores, it should not appear in the
    # verdict output and should not affect the overall verdict.
    scores = {"faithfulness": 0.95}
    result = apply_verdict(scores)
    assert "answer_relevancy" not in result["metrics"]
    assert result["overall"] == "pass"


@patch("src.validator.get_judge_llm")
@patch("src.validator.evaluate")
def test_run_validation_exports_expected_shape(mock_evaluate, mock_get_judge_llm, tmp_path):
    # get_judge_llm() would normally build a real ChatOpenAI client, which
    # requires OPENAI_API_KEY just to construct. Replace it with a plain
    # stand-in so this test never needs a real key.
    mock_get_judge_llm.return_value = MagicMock()

    # Fake RAGAS scores for 2 cases: one clean pass, one faithfulness fail
    fake_df = pd.DataFrame([
        {"faithfulness": 0.95, "answer_relevancy": 0.90,
         "context_precision": 0.85, "context_recall": 0.80},
        {"faithfulness": 0.40, "answer_relevancy": 0.90,
         "context_precision": 0.85, "context_recall": 0.80},
    ])
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = fake_df
    mock_evaluate.return_value = mock_result

    dataset_path = tmp_path / "mini_dataset.json"
    dataset_path.write_text(json.dumps([
        {
            "id": "001",
            "question": "Q1?",
            "contexts": ["ctx1"],
            "answer": "A1",
            "ground_truth": "GT1",
            "expected_verdict": "pass",
        },
        {
            "id": "002",
            "question": "Q2?",
            "contexts": ["ctx2"],
            "answer": "A2 (hallucinated)",
            "ground_truth": "GT2",
            "expected_verdict": "fail",
        },
    ]), encoding="utf-8")

    output_path = tmp_path / "results.json"
    summary = validator.run_validation(str(dataset_path), str(output_path))

    assert output_path.exists()
    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 1
    assert summary["failed_cases"] == 1
    assert summary["cases"][0]["matches_expectation"] is True
    assert summary["cases"][1]["matches_expectation"] is True

    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported == summary