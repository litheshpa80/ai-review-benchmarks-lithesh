"""Command-Line Interface for the Proofline AI Code Review Benchmark.

Usage:
    # Positional mode (combined single dataset):
    python -m benchmark.cli <path_to_dataset.json>

    # Separate-file mode (ground truth + reviews):
    python -m benchmark.cli --ground-truth <path_to_ground_truth.json> --reviews <path_to_reviews.json>
"""

import argparse
import json
from pathlib import Path
import sys

from benchmark.evaluator import evaluate_benchmark, evaluate_dataset
from benchmark.matrix import generate_markdown_matrix


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse command-line arguments and return the parsed namespace along with the parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.cli",
        description="Evaluate AI code reviewers against benchmark ground truth and generate a comparison matrix.",
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a combined JSON dataset file containing ground truth and reviewer findings.",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        dest="ground_truth",
        type=Path,
        default=None,
        help="Path to the ground-truth JSON file.",
    )
    parser.add_argument(
        "--reviews",
        "-r",
        dest="reviews",
        type=Path,
        default=None,
        help="Path to the reviewer results JSON file.",
    )
    args = parser.parse_args(argv)
    return args, parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Optional list of command-line argument strings.
              Defaults to sys.argv[1:] when None.

    Returns:
        0 on success, non-zero exit code on error.
    """
    try:
        args, parser = parse_args(argv)
    except SystemExit as err:
        return err.code if isinstance(err.code, int) else 2

    # Validation: Neither positional nor flag arguments provided
    if not args.dataset_path and not args.ground_truth and not args.reviews:
        parser.print_usage(sys.stderr)
        sys.stderr.write("Error: Must provide either a dataset path or both --ground-truth and --reviews.\n")
        return 2

    # Validation: Conflicting modes (both positional dataset and flags provided)
    if args.dataset_path and (args.ground_truth or args.reviews):
        sys.stderr.write("Error: Cannot specify both a positional dataset path and --ground-truth/--reviews.\n")
        return 1

    # Validation: Incomplete separate-file mode flags
    if (args.ground_truth and not args.reviews) or (args.reviews and not args.ground_truth):
        sys.stderr.write("Error: Both --ground-truth and --reviews must be specified together.\n")
        return 1

    # Mode 1: Separate-file mode (--ground-truth and --reviews)
    if args.ground_truth and args.reviews:
        gt_path: Path = args.ground_truth
        rev_path: Path = args.reviews

        if not gt_path.exists():
            sys.stderr.write(f"Error: File not found: {gt_path}\n")
            return 1
        if not gt_path.is_file():
            sys.stderr.write(f"Error: Path is not a file: {gt_path}\n")
            return 1

        if not rev_path.exists():
            sys.stderr.write(f"Error: File not found: {rev_path}\n")
            return 1
        if not rev_path.is_file():
            sys.stderr.write(f"Error: Path is not a file: {rev_path}\n")
            return 1

        # Validate and load ground-truth JSON
        try:
            with gt_path.open("r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except json.JSONDecodeError as err:
            sys.stderr.write(f"Error: Invalid JSON in {gt_path}: {err}\n")
            return 1

        # Validate and load reviews JSON
        try:
            with rev_path.open("r", encoding="utf-8") as f:
                rev_data = json.load(f)
        except json.JSONDecodeError as err:
            sys.stderr.write(f"Error: Invalid JSON in {rev_path}: {err}\n")
            return 1

        try:
            results = evaluate_benchmark(gt_data, rev_data)
        except Exception as err:
            sys.stderr.write(f"Error evaluating benchmark: {err}\n")
            return 1

        if not results:
            sys.stderr.write(
                "Error: No evaluation results found. Ensure input files contain valid benchmark cases and reviews.\n"
            )
            return 1

        for item in results:
            unknown = item.get("unknown_cases", [])
            if unknown:
                reviewer = item.get("reviewer", "Unknown")
                sys.stderr.write(
                    f"Warning: Reviewer '{reviewer}' submitted unknown case IDs not present in ground truth: {unknown}\n"
                )

        markdown_table = generate_markdown_matrix(results)
        print(markdown_table)
        return 0

    # Mode 2: Positional mode (combined single dataset)
    dataset_path: Path = args.dataset_path

    if not dataset_path.exists():
        sys.stderr.write(f"Error: File not found: {dataset_path}\n")
        return 1

    if not dataset_path.is_file():
        sys.stderr.write(f"Error: Path is not a file: {dataset_path}\n")
        return 1

    try:
        results = evaluate_dataset(dataset_path)
    except json.JSONDecodeError as err:
        sys.stderr.write(f"Error: Invalid JSON in {dataset_path}: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Error evaluating dataset {dataset_path}: {err}\n")
        return 1

    if not results:
        sys.stderr.write(
            f"Error: No evaluation results found in {dataset_path}. Ensure file contains valid benchmark cases and reviews.\n"
        )
        return 1

    for item in results:
        unknown = item.get("unknown_cases", [])
        if unknown:
            reviewer = item.get("reviewer", "Unknown")
            sys.stderr.write(
                f"Warning: Reviewer '{reviewer}' submitted unknown case IDs not present in ground truth: {unknown}\n"
            )

    markdown_table = generate_markdown_matrix(results)
    print(markdown_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
