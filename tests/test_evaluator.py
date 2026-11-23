"""Automated unit tests for benchmark/evaluator.py.

This test suite verifies the Proofline AI Code Review Benchmark Evaluator.
It tests:
- Core classification metrics: True Positives (TP), False Positives (FP),
  False Negatives (FN), Precision, Recall, and F1 score.
- Duplicate findings tracking (duplicates must not increase FP).
- Clean-case (control) metrics: True Negatives (TN), False Alarms (FA),
  and clean-case specificity.
- Operational status handling (completed, quota_exhausted, error, unattempted).
- Incomplete reviews must never generate False Negatives.
- Composite case-level outcomes (TP, FP, FN, TN, TP+FP, TP+FN, FN+FP, TP+FN+FP,
  QUOTA, ERROR, —).
- Case-scoped bug ID matching.
- Both single-file (combined) and separate-files (ground truth + reviews) input formats.
- Top-level {"reviews": [...]} format support.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure the project root is on sys.path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.evaluator import (
    calculate_metrics,
    calculate_specificity,
    evaluate_benchmark,
    evaluate_dataset,
)


class TestReviewEvaluator(unittest.TestCase):
    """Test suite for reviewer evaluation and metric calculations."""

    def setUp(self):
        """Prepare tracking for temporary test files."""
        self.temp_files: list[Path] = []

    def tearDown(self):
        """Clean up any temporary test files created during tests."""
        for path in self.temp_files:
            if path.exists():
                path.unlink()

    def create_dataset_file(self, data: dict | list) -> Path:
        """Helper to create a temporary JSON dataset file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, temp_file)
        temp_file.close()
        path = Path(temp_file.name)
        self.temp_files.append(path)
        return path

    # -------------------------------------------------------------------------
    # Core Scenarios
    # -------------------------------------------------------------------------

    def test_01_perfect_reviewer(self):
        """Scenario 1: Reviewer correctly matches the single ground-truth bug.

        Expected: TP=1, FP=0, FN=0, Precision=1.0, Recall=1.0, F1=1.0.
        Case outcome: 'TP'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {
                            "bug_id": "BUG-01",
                            "category": "pagination",
                            "description": "Pagination skips records.",
                        }
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {
                                    "matched_bug_id": "BUG-01",
                                    "type": "bug",
                                    "description": "Found pagination bug.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-A")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["duplicates"], 0)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(r["completed_cases"], 1)
        self.assertEqual(r["total_cases"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP"})

    def test_02_false_positive(self):
        """Scenario 2: Reviewer reports a finding with an invalid matched_bug_id.

        Expected: TP=0, FP=1, FN=1, Precision=0.0, Recall=0.0, F1=0.0.
        Case outcome: 'FN+FP' (missed real bug, claimed fake bug).
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {
                            "bug_id": "BUG-01",
                            "category": "pagination",
                            "description": "Pagination skips records.",
                        }
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {
                                    "matched_bug_id": "NON_EXISTENT_BUG",
                                    "type": "bug",
                                    "description": "Reported fake bug.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["precision"], 0.0)
        self.assertEqual(r["recall"], 0.0)
        self.assertEqual(r["f1"], 0.0)
        self.assertEqual(r["case_outcomes"], {"case-01": "FN+FP"})

    def test_03_false_negative(self):
        """Scenario 3: Reviewer reports no findings and misses the bug.

        Expected: TP=0, FP=0, FN=1, Precision=None, Recall=0.0, F1=None.
        Case outcome: 'FN'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {
                            "bug_id": "BUG-01",
                            "category": "pagination",
                            "description": "Pagination skips records.",
                        }
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 1)
        self.assertIsNone(r["precision"])
        self.assertEqual(r["recall"], 0.0)
        self.assertIsNone(r["f1"])
        self.assertEqual(r["case_outcomes"], {"case-01": "FN"})

    def test_04_duplicate_finding(self):
        """Scenario 4: Reviewer reports the same ground-truth bug twice.

        Under the approved rules:
        - The first match is a True Positive (TP=1).
        - The repeated match increments duplicate count (duplicates=1)
          and does NOT increase False Positives (FP=0).
        - Expected: TP=1, FP=0, FN=0, duplicates=1, Precision=1.0, Recall=1.0, F1=1.0.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {
                            "bug_id": "BUG-01",
                            "category": "pagination",
                            "description": "Pagination skips records.",
                        }
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {
                                    "matched_bug_id": "BUG-01",
                                    "type": "bug",
                                    "description": "First report.",
                                },
                                {
                                    "matched_bug_id": "BUG-01",
                                    "type": "bug",
                                    "description": "Duplicate report.",
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["duplicates"], 1)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP"})

    def test_05_style_finding_ignored(self):
        """Scenario 5: Reviewer reports a style finding with type='style'.

        Expected: Style findings are ignored for bug evaluation.
        TP=0, FP=0, FN=1, Precision=None, Recall=0.0, F1=None.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {
                            "bug_id": "BUG-01",
                            "category": "pagination",
                            "description": "Pagination skips records.",
                        }
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {
                                    "matched_bug_id": "BUG-01",
                                    "type": "style",
                                    "description": "Consider formatting whitespace.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 1)
        self.assertIsNone(r["precision"])
        self.assertEqual(r["recall"], 0.0)
        self.assertIsNone(r["f1"])
        self.assertEqual(r["case_outcomes"], {"case-01": "FN"})

    def test_06_incomplete_reviewer_included(self):
        """Scenario 6: Reviewers with no completed cases should still be included in results.

        Operational outcomes (ERROR, QUOTA) must be tracked, completed_cases must be 0,
        and incomplete reviews must NOT generate False Negatives (FN=0).
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "Bug 1"}
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-Error",
                            "status": "error",
                            "findings": [{"matched_bug_id": "BUG-01"}],
                        },
                        {
                            "reviewer": "Reviewer-Quota",
                            "status": "quota_exhausted",
                            "findings": [],
                        },
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 2)
        r_map = {r["reviewer"]: r for r in results}

        # Check Reviewer-Error
        r_err = r_map["Reviewer-Error"]
        self.assertEqual(r_err["completed_cases"], 0)
        self.assertEqual(r_err["total_cases"], 1)
        self.assertEqual(r_err["fn"], 0)
        self.assertEqual(r_err["tp"], 0)
        self.assertEqual(r_err["fp"], 0)
        self.assertIsNone(r_err["precision"])
        self.assertIsNone(r_err["recall"])
        self.assertIsNone(r_err["f1"])
        self.assertEqual(r_err["case_outcomes"], {"case-01": "ERROR"})

        # Check Reviewer-Quota
        r_quota = r_map["Reviewer-Quota"]
        self.assertEqual(r_quota["completed_cases"], 0)
        self.assertEqual(r_quota["total_cases"], 1)
        self.assertEqual(r_quota["fn"], 0)
        self.assertEqual(r_quota["tp"], 0)
        self.assertEqual(r_quota["fp"], 0)
        self.assertIsNone(r_quota["precision"])
        self.assertIsNone(r_quota["recall"])
        self.assertIsNone(r_quota["f1"])
        self.assertEqual(r_quota["case_outcomes"], {"case-01": "QUOTA"})

    def test_07_multiple_cases_aggregation(self):
        """Scenario 7: Same reviewer completes two cases.

        Case 1: Catches the bug (TP=1, FN=0).
        Case 2: Misses the bug (TP=0, FN=1).
        Expected Aggregated: TP=1, FP=0, FN=1, completed_cases=2, total_cases=2.
        Precision=1.0, Recall=0.5, F1=0.6667.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "Bug in case 1"}
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-01"}],
                        }
                    ],
                },
                {
                    "case_id": "case-02",
                    "ground_truth": [
                        {"bug_id": "BUG-02", "description": "Bug in case 2"}
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [],
                        }
                    ],
                },
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-A")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["completed_cases"], 2)
        self.assertEqual(r["total_cases"], 2)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 0.5)
        expected_f1 = 2 * (1.0 * 0.5) / (1.0 + 0.5)
        self.assertAlmostEqual(r["f1"], expected_f1, places=4)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP", "case-02": "FN"})

    def test_08_division_by_zero_behavior(self):
        """Scenario 8: Direct testing of calculate_metrics() division-by-zero rules.

        - No findings (TP=0, FP=0, FN=1): Precision=None, Recall=0.0, F1=None.
        - No bugs & no findings (TP=0, FP=0, FN=0): Precision=None, Recall=None, F1=None.
        - No bugs & 1 FP (TP=0, FP=1, FN=0): Precision=0.0, Recall=None, F1=None.
        - 0 TP, 1 FP, 1 FN (Precision=0.0, Recall=0.0): F1=0.0 (safe zero).
        """
        # Case A: No findings reported
        p, r, f1 = calculate_metrics(tp=0, fp=0, fn=1)
        self.assertIsNone(p)
        self.assertEqual(r, 0.0)
        self.assertIsNone(f1)

        # Case B: Clean case, no findings reported
        p, r, f1 = calculate_metrics(tp=0, fp=0, fn=0)
        self.assertIsNone(p)
        self.assertIsNone(r)
        self.assertIsNone(f1)

        # Case C: Clean case, but false positive reported
        p, r, f1 = calculate_metrics(tp=0, fp=1, fn=0)
        self.assertEqual(p, 0.0)
        self.assertIsNone(r)
        self.assertIsNone(f1)

        # Case D: Both Precision and Recall are 0.0 -> F1 should safely be 0.0
        p, r, f1 = calculate_metrics(tp=0, fp=1, fn=1)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    # -------------------------------------------------------------------------
    # Clean Case (Control Case) Tests
    # -------------------------------------------------------------------------

    def test_09_clean_case_no_findings_is_true_negative(self):
        """Scenario 9: Clean case with no defect findings yields True Negative (TN).

        Expected: clean_tn=1, clean_fa=0, clean_specificity=1.0, outcome='TN'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-07",
                    "ground_truth": [],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["clean_tn"], 1)
        self.assertEqual(r["clean_fa"], 0)
        self.assertEqual(r["clean_specificity"], 1.0)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["case_outcomes"], {"case-07": "TN"})

    def test_10_clean_case_false_finding_is_false_alarm(self):
        """Scenario 10: Clean case with a false bug finding yields False Alarm (FA).

        Expected: clean_fa=1, clean_tn=0, FP=1, clean_specificity=0.0, outcome='FP'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-07",
                    "ground_truth": [],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {
                                    "matched_bug_id": "NON_EXISTENT",
                                    "type": "bug",
                                    "description": "Hallucinated defect.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["clean_tn"], 0)
        self.assertEqual(r["clean_fa"], 1)
        self.assertEqual(r["clean_specificity"], 0.0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["case_outcomes"], {"case-07": "FP"})

    def test_11_clean_case_style_finding_ignored(self):
        """Scenario 11: Style/nit findings on clean cases are ignored and still count as TN.

        Expected: clean_tn=1, clean_fa=0, clean_specificity=1.0, FP=0, outcome='TN'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-07",
                    "ground_truth": [],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {
                                    "type": "style",
                                    "description": "Variable naming suggestion.",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["clean_tn"], 1)
        self.assertEqual(r["clean_fa"], 0)
        self.assertEqual(r["clean_specificity"], 1.0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["case_outcomes"], {"case-07": "TN"})

    # -------------------------------------------------------------------------
    # Composite Outcome Tests (TP+FN, TP+FP, TP+FN+FP)
    # -------------------------------------------------------------------------

    def test_12_composite_outcome_tp_fn(self):
        """Scenario 12: Multiple bugs where one is found and one is missed yields 'TP+FN'.

        Ground truth: BUG-01, BUG-02.
        Reviewer: finds BUG-01, misses BUG-02.
        Expected: TP=1, FN=1, FP=0, outcome='TP+FN'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "First bug"},
                        {"bug_id": "BUG-02", "description": "Second bug"},
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {"matched_bug_id": "BUG-01", "type": "bug"}
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP+FN"})

    def test_13_composite_outcome_tp_fp(self):
        """Scenario 13: All bugs found plus an extra false finding yields 'TP+FP'.

        Ground truth: BUG-01, BUG-02.
        Reviewer: finds BUG-01, finds BUG-02, reports fake BUG-FAKE.
        Expected: TP=2, FN=0, FP=1, outcome='TP+FP'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "First bug"},
                        {"bug_id": "BUG-02", "description": "Second bug"},
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {"matched_bug_id": "BUG-01", "type": "bug"},
                                {"matched_bug_id": "BUG-02", "type": "bug"},
                                {"matched_bug_id": "BUG-FAKE", "type": "bug"},
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 2)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP+FP"})

    def test_14_composite_outcome_tp_fn_fp(self):
        """Scenario 14: Partial bug detection plus a false finding yields 'TP+FN+FP'.

        Ground truth: BUG-01, BUG-02.
        Reviewer: finds BUG-01, misses BUG-02, reports fake BUG-FAKE.
        Expected: TP=1, FN=1, FP=1, outcome='TP+FN+FP'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "First bug"},
                        {"bug_id": "BUG-02", "description": "Second bug"},
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {"matched_bug_id": "BUG-01", "type": "bug"},
                                {"matched_bug_id": "BUG-FAKE", "type": "bug"},
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP+FN+FP"})

    # -------------------------------------------------------------------------
    # Operational Status Tests (QUOTA, ERROR, Unattempted)
    # -------------------------------------------------------------------------

    def test_15_quota_exhausted_reviewer_zero_fn(self):
        """Scenario 15: Reviewer with quota_exhausted status is included with zero FN.

        A quota exhaustion must NOT be penalized as a missed defect (FN=0).
        Outcome must be 'QUOTA', completed_cases=0.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "Unreviewed bug"}
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-Quota",
                            "status": "quota_exhausted",
                            "findings": [],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-Quota")
        self.assertEqual(r["completed_cases"], 0)
        self.assertEqual(r["total_cases"], 1)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["case_outcomes"], {"case-01": "QUOTA"})

    def test_16_error_reviewer_zero_fn(self):
        """Scenario 16: Reviewer with error status is included with zero FN.

        A runtime crash or API error must NOT be penalized as a missed defect (FN=0).
        Outcome must be 'ERROR', completed_cases=0.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [
                        {"bug_id": "BUG-01", "description": "Unreviewed bug"}
                    ],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-Crash",
                            "status": "error",
                            "findings": [],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-Crash")
        self.assertEqual(r["completed_cases"], 0)
        self.assertEqual(r["total_cases"], 1)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["case_outcomes"], {"case-01": "ERROR"})

    def test_17_unattempted_case_outcome_dash(self):
        """Scenario 17: Case not attempted by a reviewer yields outcome '—' with zero FN.

        When a reviewer attempts case-01 but leaves case-02 unattempted:
        - case-01 outcome is evaluated normally.
        - case-02 outcome is '—'.
        - case-02's ground-truth bug does NOT add to FN.
        - completed_cases=1, total_cases=2.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-Partial",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                        }
                    ],
                },
                {
                    "case_id": "case-02",
                    "ground_truth": [{"bug_id": "BUG-02"}],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-Partial",
                            "status": "unattempted",
                            "findings": [],
                        }
                    ],
                },
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["completed_cases"], 1)
        self.assertEqual(r["total_cases"], 2)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP", "case-02": "—"})

    # -------------------------------------------------------------------------
    # Format and Structural Tests (Separate Files, Top-level reviews, Scoping)
    # -------------------------------------------------------------------------

    def test_18_separate_ground_truth_and_reviews_files(self):
        """Scenario 18: Separate ground_truth.json and reviews.json input files.

        Verifies that evaluate_benchmark(gt_path, reviews_path) and
        evaluate_dataset(gt_path, reviews_path) load and evaluate accurately.
        """
        gt_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01", "description": "Bug 1"}],
                },
                {
                    "case_id": "case-07",
                    "ground_truth": [],
                },
            ]
        }
        reviews_data = {
            "reviews": [
                {
                    "reviewer": "Reviewer-Separate",
                    "case_id": "case-01",
                    "status": "completed",
                    "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                },
                {
                    "reviewer": "Reviewer-Separate",
                    "case_id": "case-07",
                    "status": "completed",
                    "findings": [],
                },
            ]
        }

        gt_path = self.create_dataset_file(gt_data)
        rev_path = self.create_dataset_file(reviews_data)

        # Test evaluate_benchmark()
        results = evaluate_benchmark(gt_path, rev_path)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-Separate")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["clean_tn"], 1)
        self.assertEqual(r["clean_fa"], 0)
        self.assertEqual(r["clean_specificity"], 1.0)
        self.assertEqual(r["completed_cases"], 2)
        self.assertEqual(r["total_cases"], 2)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP", "case-07": "TN"})

        # Test evaluate_dataset() with secondary path argument
        results_dataset = evaluate_dataset(gt_path, rev_path)
        self.assertEqual(results_dataset, results)

    def test_19_top_level_reviews_format(self):
        """Scenario 19: Top-level {"reviews": [...]} input format is supported.

        Validates parsing of the approved list of review objects containing
        reviewer, case_id, status, and findings.
        """
        gt_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                }
            ]
        }
        rev_data = {
            "reviews": [
                {
                    "reviewer": "Reviewer-Alpha",
                    "case_id": "case-01",
                    "status": "completed",
                    "findings": [
                        {"matched_bug_id": "BUG-01", "type": "bug"}
                    ],
                }
            ]
        }

        gt_path = self.create_dataset_file(gt_data)
        rev_path = self.create_dataset_file(rev_data)

        results = evaluate_benchmark(gt_path, rev_path)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-Alpha")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["completed_cases"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP"})

    def test_20_case_scoped_matching_prevents_cross_case_tp(self):
        """Scenario 20: matched_bug_id from another case must NOT count as a TP.

        Case-01 has BUG-01. Case-02 has BUG-02.
        Reviewer in Case-02 reports matched_bug_id: 'BUG-01'.
        Expected: BUG-01 is not a valid bug in Case-02, so it counts as FP=1, FN=1, TP=0.
        Case outcome: 'FN+FP'.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                        }
                    ],
                },
                {
                    "case_id": "case-02",
                    "ground_truth": [{"bug_id": "BUG-02"}],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            # Mistakenly citing Case-01's bug ID in Case-02
                            "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                        }
                    ],
                },
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        # Case 1 gave TP=1; Case 2 gave FP=1, FN=1
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["case_outcomes"]["case-01"], "TP")
        self.assertEqual(r["case_outcomes"]["case-02"], "FN+FP")

    def test_21_clean_specificity_none_when_no_clean_cases(self):
        """Scenario 21: clean_specificity is None when no clean cases are evaluated.

        When all evaluated cases contain ground-truth bugs:
        clean_tn=0, clean_fa=0 -> clean_specificity should be None (not 0.0 or error).
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["clean_tn"], 0)
        self.assertEqual(r["clean_fa"], 0)
        self.assertIsNone(r["clean_specificity"])

    def test_22_duplicate_findings_do_not_increase_fp(self):
        """Scenario 22: Triplicate findings matching the same bug count as 1 TP and 2 duplicates.

        Reviewer reports BUG-01 three times in the same case.
        Expected: TP=1, FP=0, duplicates=2, Precision=1.0.
        """
        data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [
                                {"matched_bug_id": "BUG-01", "type": "bug"},
                                {"matched_bug_id": "BUG-01", "type": "bug"},
                                {"matched_bug_id": "BUG-01", "type": "bug"},
                            ],
                        }
                    ],
                }
            ]
        }
        dataset_path = self.create_dataset_file(data)
        results = evaluate_dataset(dataset_path)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["duplicates"], 2)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)

    def test_23_calculate_specificity_function(self):
        """Scenario 23: Direct unit testing of calculate_specificity().

        - clean_tn=1, clean_fa=0 -> 1.0 (100% specificity)
        - clean_tn=0, clean_fa=1 -> 0.0 (0% specificity)
        - clean_tn=3, clean_fa=1 -> 0.75 (75% specificity)
        - clean_tn=0, clean_fa=0 -> None (division by zero protection)
        """
        self.assertEqual(calculate_specificity(1, 0), 1.0)
        self.assertEqual(calculate_specificity(0, 1), 0.0)
        self.assertEqual(calculate_specificity(3, 1), 0.75)
        self.assertIsNone(calculate_specificity(0, 0))

    # -------------------------------------------------------------------------
    # Nullable Collection Normalization Tests (Finding 6.1)
    # -------------------------------------------------------------------------

    def test_24_ground_truth_null_missing_and_empty(self):
        """Scenario 24: ground_truth as explicit null, missing, or empty list does not crash."""
        # 1. Explicit null ground_truth
        data_null = {"cases": [{"case_id": "case-01", "ground_truth": None}]}
        rev_data = {
            "reviews": [
                {
                    "reviewer": "Reviewer-A",
                    "case_id": "case-01",
                    "status": "completed",
                    "findings": [],
                }
            ]
        }
        res_null = evaluate_benchmark(data_null, rev_data)
        self.assertEqual(len(res_null), 1)
        self.assertEqual(res_null[0]["clean_tn"], 1)
        self.assertEqual(res_null[0]["case_outcomes"], {"case-01": "TN"})

        # 2. Missing ground_truth field
        data_missing = {"cases": [{"case_id": "case-01"}]}
        res_missing = evaluate_benchmark(data_missing, rev_data)
        self.assertEqual(len(res_missing), 1)
        self.assertEqual(res_missing[0]["clean_tn"], 1)
        self.assertEqual(res_missing[0]["case_outcomes"], {"case-01": "TN"})

        # 3. Empty list ground_truth
        data_empty = {"cases": [{"case_id": "case-01", "ground_truth": []}]}
        res_empty = evaluate_benchmark(data_empty, rev_data)
        self.assertEqual(len(res_empty), 1)
        self.assertEqual(res_empty[0]["clean_tn"], 1)
        self.assertEqual(res_empty[0]["case_outcomes"], {"case-01": "TN"})

    def test_25_findings_null_missing_and_empty(self):
        """Scenario 25: findings as explicit null, missing, or empty list does not crash."""
        gt_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                }
            ]
        }

        # 1. Explicit null findings
        rev_null = {
            "reviews": [
                {
                    "reviewer": "Reviewer-A",
                    "case_id": "case-01",
                    "status": "completed",
                    "findings": None,
                }
            ]
        }
        res_null = evaluate_benchmark(gt_data, rev_null)
        self.assertEqual(len(res_null), 1)
        self.assertEqual(res_null[0]["fn"], 1)
        self.assertEqual(res_null[0]["case_outcomes"], {"case-01": "FN"})

        # 2. Missing findings field
        rev_missing = {
            "reviews": [
                {
                    "reviewer": "Reviewer-A",
                    "case_id": "case-01",
                    "status": "completed",
                }
            ]
        }
        res_missing = evaluate_benchmark(gt_data, rev_missing)
        self.assertEqual(len(res_missing), 1)
        self.assertEqual(res_missing[0]["fn"], 1)
        self.assertEqual(res_missing[0]["case_outcomes"], {"case-01": "FN"})

        # 3. Empty list findings
        rev_empty = {
            "reviews": [
                {
                    "reviewer": "Reviewer-A",
                    "case_id": "case-01",
                    "status": "completed",
                    "findings": [],
                }
            ]
        }
        res_empty = evaluate_benchmark(gt_data, rev_empty)
        self.assertEqual(len(res_empty), 1)
        self.assertEqual(res_empty[0]["fn"], 1)
        self.assertEqual(res_empty[0]["case_outcomes"], {"case-01": "FN"})

    def test_26_case_reviewers_null_missing_and_empty(self):
        """Scenario 26: reviewers in combined format as explicit null, missing, or empty list does not crash."""
        # 1. Explicit null reviewers
        data_null = {"cases": [{"case_id": "case-01", "ground_truth": [], "reviewers": None}]}
        self.assertEqual(evaluate_dataset(data_null), [])

        # 2. Missing reviewers
        data_missing = {"cases": [{"case_id": "case-01", "ground_truth": []}]}
        self.assertEqual(evaluate_dataset(data_missing), [])

        # 3. Empty list reviewers
        data_empty = {"cases": [{"case_id": "case-01", "ground_truth": [], "reviewers": []}]}
        self.assertEqual(evaluate_dataset(data_empty), [])

    def test_27_top_level_reviewers_null_missing_and_empty(self):
        """Scenario 27: top-level reviewers as explicit null, missing, or empty list does not crash."""
        self.assertEqual(evaluate_dataset({"reviewers": None}), [])
        self.assertEqual(evaluate_dataset({}), [])
        self.assertEqual(evaluate_dataset({"reviewers": []}), [])

    def test_28_top_level_reviews_null_missing_and_empty(self):
        """Scenario 28: top-level reviews in separate reviews input as null, missing, or empty list does not crash."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": []}]}
        self.assertEqual(evaluate_benchmark(gt_data, {"reviews": None}), [])
        self.assertEqual(evaluate_benchmark(gt_data, {}), [])
        self.assertEqual(evaluate_benchmark(gt_data, {"reviews": []}), [])

    def test_29_reviewer_grouped_reviews_and_cases_null_missing_and_empty(self):
        """Scenario 29: reviews and cases in reviewer-grouped format as null, missing, or empty does not crash."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}

        # 1. Explicit null reviews
        rev_null = {"reviewers": [{"reviewer": "Reviewer-A", "reviews": None}]}
        res_null = evaluate_benchmark(gt_data, rev_null)
        self.assertEqual(len(res_null), 1)
        self.assertEqual(res_null[0]["completed_cases"], 0)
        self.assertEqual(res_null[0]["case_outcomes"], {"case-01": "—"})

        # 2. Missing reviews and cases fields
        rev_missing = {"reviewers": [{"reviewer": "Reviewer-A"}]}
        res_missing = evaluate_benchmark(gt_data, rev_missing)
        self.assertEqual(len(res_missing), 1)
        self.assertEqual(res_missing[0]["completed_cases"], 0)
        self.assertEqual(res_missing[0]["case_outcomes"], {"case-01": "—"})

        # 3. Empty list reviews
        rev_empty = {"reviewers": [{"reviewer": "Reviewer-A", "reviews": []}]}
        res_empty = evaluate_benchmark(gt_data, rev_empty)
        self.assertEqual(len(res_empty), 1)
        self.assertEqual(res_empty[0]["completed_cases"], 0)
        self.assertEqual(res_empty[0]["case_outcomes"], {"case-01": "—"})

        # 4. Explicit null cases field in reviewer object
        rev_cnull = {"reviewers": [{"reviewer": "Reviewer-A", "cases": None}]}
        res_cnull = evaluate_benchmark(gt_data, rev_cnull)
        self.assertEqual(len(res_cnull), 1)
        self.assertEqual(res_cnull[0]["completed_cases"], 0)
        self.assertEqual(res_cnull[0]["case_outcomes"], {"case-01": "—"})

    # -------------------------------------------------------------------------
    # Status Normalization Tests (Findings 4.1 & 4.2)
    # -------------------------------------------------------------------------

    def test_30_status_quota_and_quota_exhausted_display_quota(self):
        """Scenario 30: Both 'quota' and 'quota_exhausted' map to 'QUOTA' with zero FN."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}

        # 1. status: "quota"
        rev_quota = {
            "reviews": [
                {
                    "reviewer": "Reviewer-QuotaShort",
                    "case_id": "case-01",
                    "status": "quota",
                    "findings": [],
                }
            ]
        }
        res_quota = evaluate_benchmark(gt_data, rev_quota)
        self.assertEqual(len(res_quota), 1)
        self.assertEqual(res_quota[0]["case_outcomes"], {"case-01": "QUOTA"})
        self.assertEqual(res_quota[0]["completed_cases"], 0)
        self.assertEqual(res_quota[0]["fn"], 0)
        self.assertEqual(res_quota[0]["tp"], 0)
        self.assertEqual(res_quota[0]["fp"], 0)

        # 2. status: "quota_exhausted"
        rev_exhausted = {
            "reviews": [
                {
                    "reviewer": "Reviewer-QuotaLong",
                    "case_id": "case-01",
                    "status": "quota_exhausted",
                    "findings": [],
                }
            ]
        }
        res_exhausted = evaluate_benchmark(gt_data, rev_exhausted)
        self.assertEqual(len(res_exhausted), 1)
        self.assertEqual(res_exhausted[0]["case_outcomes"], {"case-01": "QUOTA"})
        self.assertEqual(res_exhausted[0]["completed_cases"], 0)
        self.assertEqual(res_exhausted[0]["fn"], 0)

    def test_31_status_case_insensitivity_and_whitespace(self):
        """Scenario 31: Status matching is case-insensitive and trims surrounding whitespace."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]},
                {"case_id": "case-02", "ground_truth": [{"bug_id": "BUG-02"}]},
                {"case_id": "case-03", "ground_truth": [{"bug_id": "BUG-03"}]},
                {"case_id": "case-04", "ground_truth": [{"bug_id": "BUG-04"}]},
                {"case_id": "case-05", "ground_truth": [{"bug_id": "BUG-05"}]},
            ]
        }

        reviews_data = {
            "reviews": [
                {
                    "reviewer": "Reviewer-Flex",
                    "case_id": "case-01",
                    "status": " Completed ",
                    "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                },
                {
                    "reviewer": "Reviewer-Flex",
                    "case_id": "case-02",
                    "status": "COMPLETED",
                    "findings": [{"matched_bug_id": "BUG-02", "type": "bug"}],
                },
                {
                    "reviewer": "Reviewer-Flex",
                    "case_id": "case-03",
                    "status": " Error ",
                    "findings": [],
                },
                {
                    "reviewer": "Reviewer-Flex",
                    "case_id": "case-04",
                    "status": "QUOTA",
                    "findings": [],
                },
                {
                    "reviewer": "Reviewer-Flex",
                    "case_id": "case-05",
                    "status": "Quota_Exhausted",
                    "findings": [],
                },
            ]
        }

        results = evaluate_benchmark(gt_data, reviews_data)
        self.assertEqual(len(results), 1)
        r = results[0]

        # Check outcomes
        self.assertEqual(r["case_outcomes"]["case-01"], "TP")
        self.assertEqual(r["case_outcomes"]["case-02"], "TP")
        self.assertEqual(r["case_outcomes"]["case-03"], "ERROR")
        self.assertEqual(r["case_outcomes"]["case-04"], "QUOTA")
        self.assertEqual(r["case_outcomes"]["case-05"], "QUOTA")

        # Only the 2 completed cases count
        self.assertEqual(r["completed_cases"], 2)
        self.assertEqual(r["total_cases"], 5)
        self.assertEqual(r["tp"], 2)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)

    def test_32_status_completed_variations_count_and_metrics(self):
        """Scenario 32: Case variations and whitespace for 'completed' compute metrics properly."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}

        for status_variant in ["completed", "Completed", "COMPLETED", "  completed  ", " Completed "]:
            with self.subTest(status=status_variant):
                rev = {
                    "reviews": [
                        {
                            "reviewer": "Rev",
                            "case_id": "case-01",
                            "status": status_variant,
                            "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                        }
                    ]
                }
                res = evaluate_benchmark(gt_data, rev)
                self.assertEqual(res[0]["completed_cases"], 1)
                self.assertEqual(res[0]["tp"], 1)
                self.assertEqual(res[0]["case_outcomes"], {"case-01": "TP"})

    def test_33_status_error_and_unattempted_remain_correct(self):
        """Scenario 33: Various formats of error, unattempted, and unknown status do not add FN."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]},
                {"case_id": "case-02", "ground_truth": [{"bug_id": "BUG-02"}]},
                {"case_id": "case-03", "ground_truth": [{"bug_id": "BUG-03"}]},
            ]
        }

        rev = {
            "reviews": [
                {
                    "reviewer": "Rev",
                    "case_id": "case-01",
                    "status": " ERROR ",
                    "findings": [],
                },
                {
                    "reviewer": "Rev",
                    "case_id": "case-02",
                    "status": " Unattempted ",
                    "findings": [],
                },
                {
                    "reviewer": "Rev",
                    "case_id": "case-03",
                    "status": "UNKNOWN_CUSTOM_STATUS",
                    "findings": [],
                },
            ]
        }
        res = evaluate_benchmark(gt_data, rev)
        self.assertEqual(res[0]["completed_cases"], 0)
        self.assertEqual(res[0]["fn"], 0)
        self.assertEqual(res[0]["case_outcomes"]["case-01"], "ERROR")
        self.assertEqual(res[0]["case_outcomes"]["case-02"], "—")
        self.assertEqual(res[0]["case_outcomes"]["case-03"], "—")

    # -------------------------------------------------------------------------
    # Raw List Format Support Tests (Finding 6.2)
    # -------------------------------------------------------------------------

    def test_34_raw_review_list_in_memory(self):
        """Scenario 34: Raw list of review records passed directly in-memory to evaluate_benchmark."""
        gt_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01", "description": "Bug 1"}],
                }
            ]
        }
        raw_reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
            }
        ]

        results = evaluate_benchmark(gt_data, raw_reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-A")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(r["completed_cases"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP"})

    def test_35_raw_review_list_from_json_file(self):
        """Scenario 35: Raw list of review records loaded from a JSON file."""
        gt_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                }
            ]
        }
        raw_reviews = [
            {
                "reviewer": "Reviewer-File",
                "case_id": "case-01",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
            }
        ]

        gt_file = self.create_dataset_file(gt_data)
        rev_file = self.create_dataset_file(raw_reviews)

        # Verify evaluate_benchmark with file path
        results = evaluate_benchmark(gt_file, rev_file)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-File")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP"})

        # Verify evaluate_dataset alias
        results_dataset = evaluate_dataset(gt_file, rev_file)
        self.assertEqual(results_dataset, results)

    def test_36_raw_review_list_reproduction_metrics_tp_fp_fn(self):
        """Scenario 36: Multi-case evaluation with raw list testing TP, FP, FN, and Precision/Recall/F1."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]},
                {"case_id": "case-02", "ground_truth": [{"bug_id": "BUG-02"}]},
            ]
        }
        raw_reviews = [
            {
                "reviewer": "Reviewer-Multi",
                "case_id": "case-01",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
            },
            {
                "reviewer": "Reviewer-Multi",
                "case_id": "case-02",
                "status": "completed",
                "findings": [{"matched_bug_id": "NON_EXISTENT", "type": "bug"}],
            },
        ]

        results = evaluate_benchmark(gt_data, raw_reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["reviewer"], "Reviewer-Multi")
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["precision"], 0.5)
        self.assertEqual(r["recall"], 0.5)
        self.assertEqual(r["f1"], 0.5)
        self.assertEqual(r["completed_cases"], 2)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP", "case-02": "FN+FP"})

    def test_37_raw_case_list_ground_truth(self):
        """Scenario 37: Ground truth provided as a raw list of cases continues to work."""
        raw_gt_list = [
            {
                "case_id": "case-01",
                "ground_truth": [{"bug_id": "BUG-01"}],
            }
        ]
        raw_reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
            }
        ]

        results = evaluate_benchmark(raw_gt_list, raw_reviews)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tp"], 1)
        self.assertEqual(results[0]["completed_cases"], 1)
        self.assertEqual(results[0]["case_outcomes"], {"case-01": "TP"})

    # -------------------------------------------------------------------------
    # Finding 3.1: Duplicate False Positives
    # -------------------------------------------------------------------------

    def test_38_duplicate_false_positive_repeated_twice(self):
        """Scenario 38: Identical false-positive finding repeated twice yields FP=1, Duplicates=1."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Hallucinated issue"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Hallucinated issue"},
                ],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["duplicates"], 1)
        self.assertEqual(r["precision"], 0.0)
        self.assertEqual(r["recall"], 0.0)
        self.assertEqual(r["f1"], 0.0)
        self.assertEqual(r["case_outcomes"], {"case-01": "FN+FP"})

    def test_39_duplicate_false_positive_repeated_four_times(self):
        """Scenario 39: Identical false-positive finding repeated 4 times yields FP=1, Duplicates=3."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Same issue"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Same issue"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Same issue"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Same issue"},
                ],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["duplicates"], 3)

    def test_40_different_false_positive_descriptions_count_separately(self):
        """Scenario 40: Two different false-positive descriptions count as FP=2, Duplicates=0."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [
                    {"type": "bug", "description": "First distinct hallucination"},
                    {"type": "bug", "description": "Second distinct hallucination"},
                ],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 2)
        self.assertEqual(r["fn"], 1)
        self.assertEqual(r["duplicates"], 0)

    def test_41_same_fake_id_different_descriptions_count_separately(self):
        """Scenario 41: Same fake matched_bug_id but different descriptions count as FP=2, Duplicates=0."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Issue on line 10"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Issue on line 45"},
                ],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 2)
        self.assertEqual(r["duplicates"], 0)

    def test_42_missing_matched_bug_id_identical_descriptions_duplicate(self):
        """Scenario 42: Missing matched_bug_id, identical descriptions (with whitespace) yield FP=1, Duplicates=1."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [
                    {"type": "bug", "description": "Whitespace test description"},
                    {"type": "bug", "description": "   Whitespace test description   "},
                ],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["duplicates"], 1)

    def test_43_same_false_positive_across_cases_independent(self):
        """Scenario 43: Same false-positive description in two different cases counts independently (FP=2, Duplicates=0)."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]},
                {"case_id": "case-02", "ground_truth": [{"bug_id": "BUG-02"}]},
            ]
        }
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [{"type": "bug", "description": "Cross case hallucination"}],
            },
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-02",
                "status": "completed",
                "findings": [{"type": "bug", "description": "Cross case hallucination"}],
            },
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 2)
        self.assertEqual(r["duplicates"], 0)
        self.assertEqual(r["case_outcomes"], {"case-01": "FN+FP", "case-02": "FN+FP"})

    def test_44_tp_and_fp_duplicates_combined(self):
        """Scenario 44: Case with both duplicate TP and duplicate FP yields TP=1, FP=1, Duplicates=2."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [
                    {"matched_bug_id": "BUG-01", "type": "bug", "description": "Found bug 1"},
                    {"matched_bug_id": "BUG-01", "type": "bug", "description": "Found bug 1 again"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Fake bug"},
                    {"matched_bug_id": "BUG-FAKE", "type": "bug", "description": "Fake bug"},
                ],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 1)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["duplicates"], 2)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP+FP"})

    # -------------------------------------------------------------------------
    # Unknown Reviewer Cases (Finding 8.3)
    # -------------------------------------------------------------------------

    def test_45_single_unknown_case_recorded(self):
        """Scenario 45: Reviewer submits single unknown case not in ground truth."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-99",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["unknown_cases"], ["case-99"])
        self.assertEqual(r["completed_cases"], 0)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["duplicates"], 0)
        self.assertEqual(r["total_cases"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "—"})

    def test_46_valid_and_unknown_cases_metrics_unchanged(self):
        """Scenario 46: Reviewer submits both valid and unknown cases; valid metrics are unaffected."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-01",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
            },
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-99",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-FAKE", "type": "bug"}],
            },
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["unknown_cases"], ["case-99"])
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["duplicates"], 0)
        self.assertEqual(r["precision"], 1.0)
        self.assertEqual(r["recall"], 1.0)
        self.assertEqual(r["f1"], 1.0)
        self.assertEqual(r["completed_cases"], 1)
        self.assertEqual(r["total_cases"], 1)
        self.assertEqual(r["case_outcomes"], {"case-01": "TP"})

    def test_47_only_unknown_cases_no_false_metrics(self):
        """Scenario 47: Reviewer submits only unknown cases; no false metrics or completed cases."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]},
                {"case_id": "case-02", "ground_truth": [{"bug_id": "BUG-02"}]},
            ]
        }
        reviews = [
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-99",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-FAKE", "type": "bug"}],
            }
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["unknown_cases"], ["case-99"])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["duplicates"], 0)
        self.assertEqual(r["completed_cases"], 0)
        self.assertEqual(r["total_cases"], 2)
        self.assertEqual(r["case_outcomes"], {"case-01": "—", "case-02": "—"})

    def test_48_multiple_reviewers_independent_unknown_cases(self):
        """Scenario 48: Multiple reviewers have independent unknown_cases lists."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        reviews = [
            {"reviewer": "Reviewer-A", "case_id": "case-01", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-A", "case_id": "case-99", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-B", "case_id": "case-01", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-B", "case_id": "case-88", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-B", "case_id": "case-77", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-C", "case_id": "case-01", "status": "completed", "findings": []},
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 3)

        results_by_reviewer = {r["reviewer"]: r for r in results}
        self.assertEqual(results_by_reviewer["Reviewer-A"]["unknown_cases"], ["case-99"])
        self.assertEqual(results_by_reviewer["Reviewer-B"]["unknown_cases"], ["case-77", "case-88"])
        self.assertEqual(results_by_reviewer["Reviewer-C"]["unknown_cases"], [])

    def test_49_no_unknown_cases_returns_empty_list(self):
        """Scenario 49: When all submitted cases match ground truth, unknown_cases is an empty list."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]},
                {"case_id": "case-02", "ground_truth": []},
            ]
        }
        reviews = [
            {"reviewer": "Reviewer-A", "case_id": "case-01", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-A", "case_id": "case-02", "status": "completed", "findings": []},
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["unknown_cases"], [])

    def test_50_unknown_cases_do_not_change_metrics_or_clean_cases(self):
        """Scenario 50: Unknown cases do not alter clean case TN/FA or bug case TP/FN/FP metrics."""
        gt_data = {
            "cases": [
                {"case_id": "case-01", "ground_truth": []},  # clean
                {"case_id": "case-02", "ground_truth": [{"bug_id": "BUG-02"}]},  # bug
            ]
        }
        reviews = [
            {"reviewer": "Reviewer-A", "case_id": "case-01", "status": "completed", "findings": []},
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-02",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-02", "type": "bug"}],
            },
            {
                "reviewer": "Reviewer-A",
                "case_id": "case-unknown",
                "status": "completed",
                "findings": [{"matched_bug_id": "BUG-FAKE", "type": "bug"}],
            },
        ]
        results = evaluate_benchmark(gt_data, reviews)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["unknown_cases"], ["case-unknown"])
        self.assertEqual(r["clean_tn"], 1)
        self.assertEqual(r["clean_fa"], 0)
        self.assertEqual(r["clean_specificity"], 1.0)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["completed_cases"], 2)
        self.assertEqual(r["total_cases"], 2)
        self.assertEqual(r["case_outcomes"], {"case-01": "TN", "case-02": "TP"})


if __name__ == "__main__":
    unittest.main()
