"""
CLI entry point for the RAG QA validator.

Usage:
    python -m src.main --dataset data/qa_dataset.json --output results.json

Exit code is 0 if every case passed its thresholds, 1 otherwise - so this can be wired into a CI step that should fail the build on regressions.
"""

import argparse
import sys

from src.validator import run_validation

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate RAG answer faithfulness against a QA dataset using RAGAS."
    )
    parser.add_argument(
        "--dataset",
        default="data/qa_dataset.json",
        help="Path to the QA dataset JSON file (default: data/qa_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to write the results JSON to (default: results.json)",
    )
    return parser

def print_summary(summary: dict) -> None:
    print(f"\nRAG QA Validator - {summary['total_cases']} cases evaluated")
    print(f"  Passed: {summary['passed_cases']}")
    print(f"  Failed: {summary['failed_cases']}\n")

    for case in summary["cases"]:
        flag = "OK" if case["matches_expectation"] else "UNEXPECTED"
        print(f"  [{case['actual_verdict'].upper():4}] {case['id']} - {case['question'][:60]} ({flag})")

    print()

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_validation(args.dataset, args.output)
    print_summary(summary)

    return 1 if summary["failed_cases"] > 0 else 0

if __name__ == "__main__":
    sys.exit(main())