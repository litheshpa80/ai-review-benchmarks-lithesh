"""Automated unit tests for benchmark/cli.py.

Tests cover:
- Positional mode:
  - Successful evaluation and Markdown report output
  - Missing input file handling
  - Invalid JSON handling
  - Path pointing to a directory instead of a file
- Separate-file mode:
  - Successful evaluation with --ground-truth and --reviews
  - Successful evaluation with short flags -g and -r
  - Incomplete flags (--ground-truth without --reviews, or vice-versa)
  - Missing ground-truth file
  - Missing reviews file
  - Directory path passed to flags
  - Invalid JSON in separate files
- General validation:
  - Missing command-line arguments (empty argv)
  - Conflicting modes (specifying both positional path and flags)
- Empty or unrecognized schemas:
  - Empty JSON object {}
  - Unrecognized JSON schema
  - Empty cases list
  - Invalid or missing case IDs
  - Empty results in separate-file mode
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

# Ensure the project root is on sys.path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.cli import main


class TestBenchmarkCLI(unittest.TestCase):
    """Test suite for the benchmark CLI module."""

    def setUp(self):
        """Prepare tracking for temporary test files."""
        self.temp_files: list[Path] = []
        self.sample_input = ROOT / "benchmark" / "sample_input.json"

    def tearDown(self):
        """Clean up any temporary test files created during tests."""
        for path in self.temp_files:
            if path.exists():
                path.unlink()

    def create_json_file(self, data: dict | list | str) -> Path:
        """Helper to create a temporary JSON file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        if isinstance(data, str):
            temp_file.write(data)
        else:
            json.dump(data, temp_file)
        temp_file.close()
        path = Path(temp_file.name)
        self.temp_files.append(path)
        return path

    # -------------------------------------------------------------------------
    # Positional Mode Tests
    # -------------------------------------------------------------------------

    def test_cli_positional_success(self):
        """Test successful execution with a valid positional dataset file."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(self.sample_input)])

        self.assertEqual(exit_code, 0)
        output = stdout_capture.getvalue()
        self.assertIn("## Reviewer-by-Case Matrix", output)
        self.assertIn("## Reviewer Metrics Summary", output)
        self.assertIn(
            "| Reviewer | TP | FP | FN | Duplicates | Precision | Recall | F1 | "
            "Clean TN | Clean FA | Specificity | Completed |",
            output,
        )
        self.assertIn("| Reviewer-A | TP | 1/1 |", output)
        self.assertIn(
            "| Reviewer-A | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 0 | 0 | N/A | 1/1 |",
            output,
        )
        self.assertEqual(stderr_capture.getvalue(), "")

    def test_cli_positional_missing_file(self):
        """Test graceful handling when the positional target file does not exist."""
        non_existent = ROOT / "benchmark" / "does_not_exist_12345.json"
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(non_existent)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: File not found", stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_positional_invalid_json(self):
        """Test graceful handling when positional file is not valid JSON."""
        path = self.create_json_file("{ this is not valid json")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Invalid JSON", stderr_capture.getvalue())
        self.assertIn(f"Error: Invalid JSON in {path}:", stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_positional_directory_path(self):
        """Test graceful handling when positional path points to a directory, not a file."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(ROOT / "benchmark")])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Path is not a file", stderr_capture.getvalue())

    # -------------------------------------------------------------------------
    # Separate-File Mode Tests (--ground-truth and --reviews)
    # -------------------------------------------------------------------------

    def test_cli_separate_files_success(self):
        """Test successful execution with separate --ground-truth and --reviews files."""
        gt_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
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

        gt_file = self.create_json_file(gt_data)
        rev_file = self.create_json_file(reviews_data)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([
                "--ground-truth", str(gt_file),
                "--reviews", str(rev_file),
            ])

        self.assertEqual(exit_code, 0)
        output = stdout_capture.getvalue()
        self.assertIn("## Reviewer-by-Case Matrix", output)
        self.assertIn("## Reviewer Metrics Summary", output)
        self.assertIn("| Reviewer-Separate | TP | TN | 2/2 |", output)
        self.assertIn(
            "| Reviewer-Separate | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1 | 0 | 100.0% | 2/2 |",
            output,
        )
        self.assertEqual(stderr_capture.getvalue(), "")

    def test_cli_separate_files_short_flags(self):
        """Test separate-file mode using short flags -g and -r."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        rev_data = {
            "reviews": [
                {
                    "reviewer": "Reviewer-Short",
                    "case_id": "case-01",
                    "status": "completed",
                    "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                }
            ]
        }

        gt_file = self.create_json_file(gt_data)
        rev_file = self.create_json_file(rev_data)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["-g", str(gt_file), "-r", str(rev_file)])

        self.assertEqual(exit_code, 0)
        output = stdout_capture.getvalue()
        self.assertIn("| Reviewer-Short | TP | 1/1 |", output)
        self.assertEqual(stderr_capture.getvalue(), "")

    def test_cli_separate_files_missing_reviews_flag(self):
        """Test error when --ground-truth is provided without --reviews."""
        gt_file = self.create_json_file({"cases": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Error: Both --ground-truth and --reviews must be specified together.",
            stderr_capture.getvalue(),
        )

    def test_cli_separate_files_missing_ground_truth_flag(self):
        """Test error when --reviews is provided without --ground-truth."""
        rev_file = self.create_json_file({"reviews": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Error: Both --ground-truth and --reviews must be specified together.",
            stderr_capture.getvalue(),
        )

    def test_cli_separate_files_gt_not_found(self):
        """Test error when --ground-truth points to a non-existent file."""
        non_existent = ROOT / "benchmark" / "non_existent_gt.json"
        rev_file = self.create_json_file({"reviews": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(non_existent), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: File not found", stderr_capture.getvalue())

    def test_cli_separate_files_reviews_not_found(self):
        """Test error when --reviews points to a non-existent file."""
        gt_file = self.create_json_file({"cases": []})
        non_existent = ROOT / "benchmark" / "non_existent_reviews.json"

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(non_existent)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: File not found", stderr_capture.getvalue())

    def test_cli_separate_files_directory_path(self):
        """Test error when --ground-truth points to a directory instead of a file."""
        rev_file = self.create_json_file({"reviews": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(ROOT / "benchmark"), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Path is not a file", stderr_capture.getvalue())

    def test_cli_separate_files_invalid_json(self):
        """Test error when the ground-truth file contains invalid JSON."""
        gt_file = self.create_json_file("{ invalid json")
        rev_file = self.create_json_file({"reviews": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Invalid JSON", stderr_capture.getvalue())
        self.assertIn(f"Error: Invalid JSON in {gt_file}:", stderr_capture.getvalue())
        self.assertNotIn(str(rev_file), stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_separate_files_reviews_invalid_json(self):
        """Test error when the reviews file contains invalid JSON."""
        gt_file = self.create_json_file({"cases": []})
        rev_file = self.create_json_file("{ invalid json in reviews")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: Invalid JSON", stderr_capture.getvalue())
        self.assertIn(f"Error: Invalid JSON in {rev_file}:", stderr_capture.getvalue())
        self.assertNotIn(str(gt_file), stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    # -------------------------------------------------------------------------
    # General Validation Tests
    # -------------------------------------------------------------------------

    def test_cli_missing_arguments(self):
        """Test graceful handling when no dataset arguments are provided."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("usage:", stderr_capture.getvalue().lower())

    def test_cli_conflicting_modes(self):
        """Test error when both positional dataset path and separate flags are specified."""
        gt_file = self.create_json_file({"cases": []})
        rev_file = self.create_json_file({"reviews": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([
                str(self.sample_input),
                "--ground-truth", str(gt_file),
                "--reviews", str(rev_file),
            ])

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Error: Cannot specify both a positional dataset path and --ground-truth/--reviews.",
            stderr_capture.getvalue(),
        )

    # -------------------------------------------------------------------------
    # Finding 7.2 Regression Tests (Empty or Unrecognized Input Schemas)
    # -------------------------------------------------------------------------

    def test_cli_positional_empty_json_object(self):
        """Test that an empty JSON object {} in positional mode exits with code 1 and error in stderr."""
        path = self.create_json_file({})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No evaluation results found", stderr_capture.getvalue())
        self.assertIn(str(path), stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_positional_unrecognized_schema(self):
        """Test that an unrecognized JSON schema in positional mode exits with code 1 and error in stderr."""
        path = self.create_json_file({"hello": "world", "status": "unknown"})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No evaluation results found", stderr_capture.getvalue())
        self.assertIn(str(path), stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_positional_empty_cases_list(self):
        """Test that a dataset with an empty cases list exits with code 1 and error in stderr."""
        path = self.create_json_file({"cases": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No evaluation results found", stderr_capture.getvalue())
        self.assertIn(str(path), stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_positional_invalid_or_missing_case_ids(self):
        """Test that cases missing valid case_id fields result in exit code 1 and error in stderr."""
        path = self.create_json_file({
            "cases": [
                {"ground_truth": []},  # missing case_id
                {"case_id": None, "ground_truth": []},  # null case_id
                {"case_id": "", "ground_truth": []},  # empty string case_id
            ]
        })

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No evaluation results found", stderr_capture.getvalue())
        self.assertIn(str(path), stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_separate_files_empty_results(self):
        """Test that separate files resulting in no evaluation results exit with code 1 and error in stderr."""
        gt_file = self.create_json_file({"cases": [{"case_id": "case-01", "ground_truth": []}]})
        rev_file = self.create_json_file({"reviews": []})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No evaluation results found", stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    def test_cli_separate_files_unrecognized_schema(self):
        """Test that unrecognized schemas in separate-file mode exit with code 1 and error in stderr."""
        gt_file = self.create_json_file({"unknown_gt": 123})
        rev_file = self.create_json_file({"unknown_reviews": 456})

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: No evaluation results found", stderr_capture.getvalue())
        self.assertEqual(stdout_capture.getvalue(), "")

    # -------------------------------------------------------------------------
    # Unknown Case ID Warning Tests (Finding 8.3)
    # -------------------------------------------------------------------------

    def test_cli_positional_unknown_case_warning(self):
        """Test that positional mode warns on stderr when reviewer submits unknown cases and exits 0."""
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
                    "case_id": "case-99",  # not in any ground truth definition
                    "reviewers": [
                        {
                            "reviewer": "Reviewer-A",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-FAKE", "type": "bug"}],
                        }
                    ],
                },
            ]
        }
        # In a combined dataset where case-99 has no ground_truth, case-99 is treated as a clean case (0 bugs).
        # To test truly unknown cases not in ground truth at all, test with separate ground_truth and reviewer cases.
        # But we can also test positional mode with raw reviewer records:
        data_positional = {
            "cases": [
                {
                    "case_id": "case-01",
                    "ground_truth": [{"bug_id": "BUG-01"}],
                }
            ],
            "reviewers": [
                {
                    "reviewer": "Reviewer-A",
                    "cases": [
                        {
                            "case_id": "case-01",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                        },
                        {
                            "case_id": "case-99",
                            "status": "completed",
                            "findings": [{"matched_bug_id": "BUG-FAKE", "type": "bug"}],
                        },
                    ],
                }
            ],
        }
        path = self.create_json_file(data_positional)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 0)
        stderr_output = stderr_capture.getvalue()
        self.assertIn("Warning: Reviewer 'Reviewer-A' submitted unknown case IDs not present in ground truth: ['case-99']", stderr_output)
        stdout_output = stdout_capture.getvalue()
        self.assertIn("| Reviewer | case-01 | Completed |", stdout_output)
        self.assertNotIn("case-99", stdout_output)

    def test_cli_separate_files_unknown_case_warning(self):
        """Test that separate-file mode warns on stderr for unknown cases and exits 0 with matrix."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        rev_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "reviewer": "Reviewer-A",
                    "status": "completed",
                    "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                },
                {
                    "case_id": "case-99",
                    "reviewer": "Reviewer-A",
                    "status": "completed",
                    "findings": [{"matched_bug_id": "BUG-FAKE", "type": "bug"}],
                },
            ]
        }
        gt_file = self.create_json_file(gt_data)
        rev_file = self.create_json_file(rev_data)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 0)
        stderr_output = stderr_capture.getvalue()
        self.assertIn("Warning: Reviewer 'Reviewer-A' submitted unknown case IDs not present in ground truth: ['case-99']", stderr_output)
        stdout_output = stdout_capture.getvalue()
        self.assertIn("| Reviewer | case-01 | Completed |", stdout_output)
        self.assertNotIn("case-99", stdout_output)

    def test_cli_no_unknown_cases_produces_no_warning(self):
        """Test that clean evaluations produce zero warnings on stderr."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        rev_data = {
            "cases": [
                {
                    "case_id": "case-01",
                    "reviewer": "Reviewer-A",
                    "status": "completed",
                    "findings": [{"matched_bug_id": "BUG-01", "type": "bug"}],
                }
            ]
        }
        gt_file = self.create_json_file(gt_data)
        rev_file = self.create_json_file(rev_data)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_capture.getvalue(), "")
        self.assertIn("| Reviewer | case-01 | Completed |", stdout_capture.getvalue())

    def test_cli_multiple_reviewers_independent_warnings(self):
        """Test that multiple reviewers generate separate warnings only when they have unknown cases."""
        gt_data = {"cases": [{"case_id": "case-01", "ground_truth": [{"bug_id": "BUG-01"}]}]}
        rev_data = [
            {"reviewer": "Reviewer-A", "case_id": "case-01", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-A", "case_id": "case-99", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-B", "case_id": "case-01", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-C", "case_id": "case-01", "status": "completed", "findings": []},
            {"reviewer": "Reviewer-C", "case_id": "case-88", "status": "completed", "findings": []},
        ]
        gt_file = self.create_json_file(gt_data)
        rev_file = self.create_json_file(rev_data)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exit_code = main(["--ground-truth", str(gt_file), "--reviews", str(rev_file)])

        self.assertEqual(exit_code, 0)
        stderr_output = stderr_capture.getvalue()
        self.assertIn("Warning: Reviewer 'Reviewer-A' submitted unknown case IDs not present in ground truth: ['case-99']", stderr_output)
        self.assertNotIn("Reviewer-B", stderr_output)
        self.assertIn("Warning: Reviewer 'Reviewer-C' submitted unknown case IDs not present in ground truth: ['case-88']", stderr_output)


if __name__ == "__main__":
    unittest.main()
