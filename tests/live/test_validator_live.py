"""
Live tests for the RAG QA validator - makes real OpenAI calls through RAGAS.

Only meant to run via the live-ci.yml workflow (workflow_dispatch) or manually with OPENAI_API_KEY set. Deliberately excluded from mock-ci.yml to avoid incurring API cost on every push.

Uses the model pinned in src/metrics.py (gpt-4o-mini) to keep a full run of the 6-case dataset to a few cents.

Run manually:
    pytest tests/live/test_validator_live.py --alluredir=allure-results
"""

import os

import allure
import pytest

from src.validator import run_validation

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping live RAGAS evaluation",
)


@allure.feature("RAG Faithfulness Validation")
@allure.story("Full dataset live evaluation")
def test_full_dataset_against_taskflow_docs():
    with allure.step("Run validator against data/qa_dataset.json with live RAGAS metrics"):
        summary = run_validation(
            dataset_path="data/qa_dataset.json",
            output_path="results.json",
        )

    with allure.step("Attach results.json to the Allure report"):
        allure.attach.file(
            "results.json",
            name="results.json",
            attachment_type=allure.attachment_type.JSON,
        )

    with allure.step("Attach per-case metric breakdown"):
        for case in summary["cases"]:
            allure.attach(
                str(case["metrics"]),
                name=f"case {case['id']} - {case['question'][:40]}",
                attachment_type=allure.attachment_type.JSON,
            )

    with allure.step("Check every case's verdict matches its expected_verdict"):
        mismatches = [
            case for case in summary["cases"] if not case["matches_expectation"]
        ]
        assert not mismatches, (
            f"{len(mismatches)} case(s) did not match their expected_verdict: "
            f"{[c['id'] for c in mismatches]}"
        )