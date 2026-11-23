"""Automated unit tests for benchmark/matrix.py.

This test suite verifies the Markdown report generator for the Proofline AI Code Review
Benchmark. It tests:
- Two-section report structure: Reviewer-by-Case Matrix and Reviewer Metrics Summary.
- Dynamic case columns extracted from case_outcomes and naturally sorted.
- All 12 summary columns: Reviewer, TP, FP, FN, Duplicates, Precision, Recall, F1,
  Clean TN, Clean FA, Specificity, Completed.
- Percentage formatting with one decimal place (e.g. 83.3%, 100.0%).
- N/A formatting for None metrics.
- Completed cases formatted as 'completed_cases/total_cases'.
- Case outcome rendering: TP, FP, FN, TN, TP+FP, TP+FN, FN+FP, TP+FN+FP, QUOTA, ERROR, —.
- Graceful empty results handling.
- Safe fallbacks for missing dictionary keys.
"""

from pathlib import Path
import sys
import unittest

# Ensure the project root is on sys.path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.matrix import generate_markdown_matrix


class TestBenchmarkMatrix(unittest.TestCase):
    """Test suite for Markdown matrix and report generation."""

    def test_01_empty_results(self):
        """Test that an empty results list is handled gracefully."""
        output = generate_markdown_matrix([])
        self.assertEqual(output, "No evaluation results to display.")

    def test_02_two_section_structure_headers(self):
        """Test that the generated report contains both required Markdown section headers."""
        results = [
            {
                "reviewer": "Reviewer-A",
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "duplicates": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "clean_tn": 0,
                "clean_fa": 0,
                "clean_specificity": None,
                "completed_cases": 1,
                "total_cases": 1,
                "case_outcomes": {"case-01": "TP"},
            }
        ]

        output = generate_markdown_matrix(results)
        self.assertIn("## Reviewer-by-Case Matrix", output)
        self.assertIn("## Reviewer Metrics Summary", output)

    def test_03_reviewer_metrics_summary_columns_and_formatting(self):
        """Test that Section 2 displays all 12 columns with properly formatted metrics."""
        results = [
            {
                "reviewer": "CodeRabbit",
                "tp": 5,
                "fp": 1,
                "fn": 2,
                "duplicates": 1,
                "precision": 0.8333,
                "recall": 0.7143,
                "f1": 0.7692,
                "clean_tn": 1,
                "clean_fa": 0,
                "clean_specificity": 1.0,
                "completed_cases": 8,
                "total_cases": 8,
                "case_outcomes": {"case-01": "TP"},
            }
        ]

        output = generate_markdown_matrix(results)
        expected_header = (
            "| Reviewer | TP | FP | FN | Duplicates | Precision | Recall | F1 | "
            "Clean TN | Clean FA | Specificity | Completed |"
        )
        expected_row = (
            "| CodeRabbit | 5 | 1 | 2 | 1 | 83.3% | 71.4% | 76.9% | 1 | 0 | 100.0% | 8/8 |"
        )

        self.assertIn(expected_header, output)
        self.assertIn(expected_row, output)

    def test_04_none_metrics_display_na(self):
        """Test that None values for precision, recall, f1, and specificity are formatted as 'N/A'."""
        results = [
            {
                "reviewer": "SilentReviewer",
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "duplicates": 0,
                "precision": None,
                "recall": 0.0,
                "f1": None,
                "clean_tn": 0,
                "clean_fa": 0,
                "clean_specificity": None,
                "completed_cases": 1,
                "total_cases": 2,
                "case_outcomes": {"case-01": "FN", "case-02": "—"},
            }
        ]

        output = generate_markdown_matrix(results)
        expected_row = (
            "| SilentReviewer | 0 | 0 | 1 | 0 | N/A | 0.0% | N/A | 0 | 0 | N/A | 1/2 |"
        )
        self.assertIn(expected_row, output)

    def test_05_dynamic_case_columns_natural_sort(self):
        """Test that case columns in Section 1 are dynamically generated and naturally sorted."""
        results = [
            {
                "reviewer": "Reviewer-A",
                "completed_cases": 4,
                "total_cases": 4,
                "case_outcomes": {
                    "case-03": "TP+FN",
                    "case-01": "TP",
                    "case-07": "TN",
                    "case-02": "FN+FP",
                },
            }
        ]

        output = generate_markdown_matrix(results)
        # Expected naturally sorted column order: case-01, case-02, case-03, case-07
        expected_case_header = "| Reviewer | case-01 | case-02 | case-03 | case-07 | Completed |"
        expected_case_row = "| Reviewer-A | TP | FN+FP | TP+FN | TN | 4/4 |"

        self.assertIn(expected_case_header, output)
        self.assertIn(expected_case_row, output)

    def test_06_case_outcomes_all_types(self):
        """Test that Section 1 accurately reflects diverse composite and operational outcomes."""
        results = [
            {
                "reviewer": "Reviewer-A",
                "completed_cases": 4,
                "total_cases": 4,
                "case_outcomes": {
                    "case-01": "TP",
                    "case-02": "FN+FP",
                    "case-03": "TP+FN",
                    "case-07": "TN",
                },
            },
            {
                "reviewer": "Reviewer-B",
                "completed_cases": 2,
                "total_cases": 4,
                "case_outcomes": {
                    "case-01": "TP",
                    "case-02": "QUOTA",
                    "case-03": "ERROR",
                    "case-07": "TN",
                },
            },
            {
                "reviewer": "Reviewer-C",
                "completed_cases": 3,
                "total_cases": 4,
                "case_outcomes": {
                    "case-01": "TP+FP",
                    "case-02": "TP+FN+FP",
                    "case-03": "FP",
                    "case-07": "—",
                },
            },
        ]

        output = generate_markdown_matrix(results)
        self.assertIn("| Reviewer-A | TP | FN+FP | TP+FN | TN | 4/4 |", output)
        self.assertIn("| Reviewer-B | TP | QUOTA | ERROR | TN | 2/4 |", output)
        self.assertIn("| Reviewer-C | TP+FP | TP+FN+FP | FP | — | 3/4 |", output)

    def test_07_unattempted_case_omitted_from_outcomes_renders_dash(self):
        """Test that if a case is absent from a reviewer's case_outcomes dict, it renders '—'."""
        results = [
            {
                "reviewer": "Reviewer-Partial",
                "completed_cases": 1,
                "total_cases": 2,
                "case_outcomes": {
                    "case-01": "TP",
                    # case-02 is omitted
                },
            },
            {
                "reviewer": "Reviewer-Full",
                "completed_cases": 2,
                "total_cases": 2,
                "case_outcomes": {
                    "case-01": "TP",
                    "case-02": "TN",
                },
            },
        ]

        output = generate_markdown_matrix(results)
        # Columns include case-01, case-02
        self.assertIn("| Reviewer | case-01 | case-02 | Completed |", output)
        # Reviewer-Partial should show '—' for missing case-02
        self.assertIn("| Reviewer-Partial | TP | — | 1/2 |", output)
        self.assertIn("| Reviewer-Full | TP | TN | 2/2 |", output)

    def test_08_fallback_completed_when_keys_missing(self):
        """Test graceful fallback for completed string when completed_cases or total_cases are absent."""
        results = [
            {
                "reviewer": "LegacyFormat",
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "cases_evaluated": 3,
                # completed_cases and total_cases not provided
            }
        ]

        output = generate_markdown_matrix(results)
        # Should fall back safely to 3/3 without throwing KeyError
        self.assertIn("3/3", output)

    def test_09_multiple_reviewers_in_both_tables(self):
        """Test that multiple reviewers are rendered in both sections."""
        results = [
            {
                "reviewer": "Reviewer-A",
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "duplicates": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "completed_cases": 1,
                "total_cases": 1,
                "case_outcomes": {"case-01": "TP"},
            },
            {
                "reviewer": "Reviewer-B",
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "duplicates": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "completed_cases": 1,
                "total_cases": 1,
                "case_outcomes": {"case-01": "FN+FP"},
            },
        ]

        output = generate_markdown_matrix(results)
        # Section 1 checks
        self.assertIn("| Reviewer-A | TP | 1/1 |", output)
        self.assertIn("| Reviewer-B | FN+FP | 1/1 |", output)

        # Section 2 checks
        self.assertIn(
            "| Reviewer-A | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0 | 0 | N/A | 1/1 |",
            output,
        )
        self.assertIn(
            "| Reviewer-B | 0 | 1 | 1 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 | N/A | 1/1 |",
            output,
        )

    def test_10_numeric_case_ids_sorting_and_outcomes(self):
        """Test that integer case IDs (1, 2, 10) are naturally sorted and retrieve outcomes."""
        results = [
            {
                "reviewer": "Reviewer-A",
                "completed_cases": 3,
                "total_cases": 3,
                "case_outcomes": {
                    10: "TN",
                    1: "TP",
                    2: "FP",
                },
            }
        ]
        output = generate_markdown_matrix(results)
        expected_case_header = "| Reviewer | 1 | 2 | 10 | Completed |"
        expected_case_row = "| Reviewer-A | TP | FP | TN | 3/3 |"
        self.assertIn(expected_case_header, output)
        self.assertIn(expected_case_row, output)

    def test_11_mixed_integer_and_string_case_ids(self):
        """Test that mixed integer and string case IDs sort naturally without crashing."""
        results = [
            {
                "reviewer": "Reviewer-Mixed",
                "completed_cases": 4,
                "total_cases": 4,
                "case_outcomes": {
                    1: "TP",
                    "case-02": "TN",
                    10: "FP",
                    "case-1": "FN",
                },
            }
        ]
        output = generate_markdown_matrix(results)
        expected_case_header = "| Reviewer | 1 | 10 | case-1 | case-02 | Completed |"
        expected_case_row = "| Reviewer-Mixed | TP | FP | FN | TN | 4/4 |"
        self.assertIn(expected_case_header, output)
        self.assertIn(expected_case_row, output)

    def test_12_numeric_case_ids_markdown_headers_and_type_coercion(self):
        """Test that Markdown headers and rows render correctly with int and str keys and missing outcomes."""
        results = [
            {
                "reviewer": "Reviewer-IntKeys",
                "completed_cases": 2,
                "total_cases": 3,
                "case_outcomes": {
                    1: "TP",
                    2: "FP",
                    # 10 is omitted
                },
            },
            {
                "reviewer": "Reviewer-StrKeys",
                "completed_cases": 3,
                "total_cases": 3,
                "case_outcomes": {
                    "1": "TN",
                    "2": "TP",
                    "10": "QUOTA",
                },
            },
        ]
        output = generate_markdown_matrix(results)
        expected_case_header = "| Reviewer | 1 | 2 | 10 | Completed |"
        expected_row_int = "| Reviewer-IntKeys | TP | FP | — | 2/3 |"
        expected_row_str = "| Reviewer-StrKeys | TN | TP | QUOTA | 3/3 |"
        self.assertIn(expected_case_header, output)
        self.assertIn(expected_row_int, output)
        self.assertIn(expected_row_str, output)

    def test_13_direct_natural_sort_key_with_integers(self):
        """Test _natural_sort_key directly with integer, string, and mixed inputs."""
        from benchmark.matrix import _natural_sort_key

        self.assertEqual(_natural_sort_key(1), ["", 1, ""])
        self.assertEqual(_natural_sort_key(10), ["", 10, ""])
        self.assertEqual(_natural_sort_key("case-02"), ["case-", 2, ""])

        raw_keys = [10, 2, 1, "case-10", "case-2"]
        sorted_keys = sorted(raw_keys, key=_natural_sort_key)
        self.assertEqual(sorted_keys, [1, 2, 10, "case-2", "case-10"])


if __name__ == "__main__":
    unittest.main()
