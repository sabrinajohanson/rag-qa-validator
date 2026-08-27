"""
Metric configuration and verdict logic for RAG faithfulness validation.

Wraps the four RAGAS metrics used to evaluate whether TaskFlow API QA answers stay faithful to the documentation they were retrieved from.

Thresholds were calibrated against the qa_dataset.json ground truth cases.
See README.md > Findings for the story behind the faithfulness threshold recalibration (0.90 -> 0.80).
"""

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# Explicitly pinned to a cheap, deterministic model so the live suite never
# silently falls back to a more expensive default. See README > Running
# Locally for the cost note.
EVAL_MODEL = "gpt-4o-mini"

def get_judge_llm() -> LangchainLLMWrapper:
    """
    Build the LLM used internally by RAGAS to score each metric.

    Wrapped explicitly (instead of relying on RAGAS's default) so the model
    and cost are pinned and visible in code, not implicit.
    """
    return LangchainLLMWrapper(ChatOpenAI(model=EVAL_MODEL, temperature=0))

# Metric instances passed to ragas.evaluate()
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# Pass/fail thresholds per metric (0.0-1.0 scale)
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.70,
}

def apply_verdict(scores: dict) -> dict:
    """
    Apply per-metric thresholds to a row of RAGAS scores and compute an overall verdict.

    Args:
        scores: dict mapping metric name -> float score, e.g. {"faithfulness": 0.62, "answer_relevancy": 0.88, ...}

    Returns:
        dict with per-metric score/threshold/verdict, plus an "overall" verdict that is "fail" if any evaluated metric fails.
    """
    metric_verdicts = {}
    overall_pass = True

    for metric_name, threshold in THRESHOLDS.items():
        score = scores.get(metric_name)
        if score is None:
            continue
        passed = score >= threshold
        metric_verdicts[metric_name] = {
            "score": round(score, 4),
            "threshold": threshold,
            "verdict": "pass" if passed else "fail",
        }
        if not passed:
            overall_pass = False

    return {
        "metrics": metric_verdicts,
        "overall": "pass" if overall_pass else "fail",
    }