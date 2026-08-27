"""
Orchestrates the RAG faithfulness validation flow: load dataset -> run RAGAS evaluation -> apply verdicts -> export results.json

Follows the same results.json export pattern used across the other AI QA repos (llm-test-case-generator, llm-as-judge) for consistency.

Note on RAGAS field names: this targets the RAGAS >=0.2 sample schema (user_input / retrieved_contexts / response / reference). If a different RAGAS version is installed and field names have changed, adjust SingleTurnSample construction in load_dataset() accordingly.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

from ragas import evaluate, EvaluationDataset, SingleTurnSample

from src.metrics import METRICS, apply_verdict, get_judge_llm

def load_dataset(dataset_path: str) -> tuple[EvaluationDataset, list[dict]]:
    """
    Load the QA dataset JSON and convert it into a RAGAS EvaluationDataset.

    Expected JSON shape (data/qa_dataset.json):
        [
          {
            "id": "001",
            "question": "...",
            "contexts": ["..."],
            "answer": "...",
            "ground_truth": "...",
            "expected_verdict": "pass" | "fail",
            "note": "..."
          },
          ...
        ]

    Returns:
        Tuple of (EvaluationDataset for ragas.evaluate, raw list of case dicts kept around for reporting/export).
    """
    raw_cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))

    samples = [
        SingleTurnSample(
            user_input=case["question"],
            retrieved_contexts=case["contexts"],
            response=case["answer"],
            reference=case["ground_truth"],
        )
        for case in raw_cases
    ]

    return EvaluationDataset(samples=samples), raw_cases

def run_validation(dataset_path: str, output_path: str) -> dict:
    """
    Run the full validation flow and write results.json to output_path.

    Returns the summary dict that was written, so callers (main.py, tests) can inspect it without re-reading the file from disk.
    """
    eval_dataset, raw_cases = load_dataset(dataset_path)

    result = evaluate(dataset=eval_dataset, metrics=METRICS, llm=get_judge_llm())
    scores_df = result.to_pandas()

    case_results = []
    fail_count = 0

    for i, case in enumerate(raw_cases):
        row = scores_df.iloc[i]
        scores = {
            "faithfulness": float(row.get("faithfulness", float("nan"))),
            "answer_relevancy": float(row.get("answer_relevancy", float("nan"))),
            "context_precision": float(row.get("context_precision", float("nan"))),
            "context_recall": float(row.get("context_recall", float("nan"))),
        }
        verdict = apply_verdict(scores)

        if verdict["overall"] == "fail":
            fail_count += 1

        case_results.append({
            "id": case["id"],
            "question": case["question"],
            "expected_verdict": case.get("expected_verdict"),
            "actual_verdict": verdict["overall"],
            "matches_expectation": verdict["overall"] == case.get("expected_verdict"),
            "metrics": verdict["metrics"],
            "note": case.get("note", ""),
        })

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path,
        "total_cases": len(raw_cases),
        "failed_cases": fail_count,
        "passed_cases": len(raw_cases) - fail_count,
        "cases": case_results,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary