"""Benchmark Matrix Generator for Proofline AI Code Review Benchmark.

This module formats evaluation results from benchmark.evaluator into
a structured Markdown report containing:
1. Reviewer-by-Case Matrix (displaying case-level outcomes dynamically per case).
2. Reviewer Metrics Summary (displaying aggregated classification and control metrics).
"""

import re
from typing import Any


def _format_percentage(value: float | None) -> str:
    """Format a metric value as a percentage with one decimal place, or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _natural_sort_key(s: Any) -> list[int | str]:
    """Sort key for alphanumeric strings or numbers so 'case-2' comes before 'case-10' and 2 comes before 10."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def _format_completed(item: dict[str, Any], default_total: int = 0) -> str:
    """Format completed cases as 'completed/total'."""
    completed = item.get("completed_cases")
    if completed is None:
        completed = item.get("cases_evaluated")

    total = item.get("total_cases")
    if total is None:
        total = default_total if default_total > 0 else completed

    if completed is None and total is None:
        return "N/A"

    c_val = completed if completed is not None else 0
    t_val = total if total is not None else c_val
    return f"{c_val}/{t_val}"


def generate_markdown_matrix(results: list[dict[str, Any]]) -> str:
    """Generate a Markdown benchmark report with case matrix and summary tables.

    Section 1: Reviewer-by-Case Matrix
        Reviewer | <case-01> | <case-02> | ... | Completed

    Section 2: Reviewer Metrics Summary
        Reviewer | TP | FP | FN | Duplicates | Precision | Recall | F1 | Clean TN | Clean FA | Specificity | Completed

    Args:
        results: A list of dictionaries containing evaluation metrics per reviewer.

    Returns:
        A formatted Markdown report string, or a message if results are empty.
    """
    if not results:
        return "No evaluation results to display."

    # Extract all unique case IDs dynamically across reviewers
    seen_cases: set[str] = set()
    for item in results:
        outcomes = item.get("case_outcomes", {})
        if isinstance(outcomes, dict):
            for cid in outcomes:
                seen_cases.add(str(cid))

    sorted_case_ids = sorted(seen_cases, key=_natural_sort_key)
    num_cases = len(sorted_case_ids)

    # -------------------------------------------------------------------------
    # Section 1: Reviewer-by-Case Matrix
    # -------------------------------------------------------------------------
    case_headers = ["Reviewer"] + sorted_case_ids + ["Completed"]
    case_separator = ["---"] * len(case_headers)

    case_lines = [
        "## Reviewer-by-Case Matrix",
        "",
        "| " + " | ".join(case_headers) + " |",
        "| " + " | ".join(case_separator) + " |",
    ]

    for item in results:
        reviewer = str(item.get("reviewer", "Unknown"))
        outcomes = item.get("case_outcomes", {})
        if not isinstance(outcomes, dict):
            outcomes = {}

        row = [reviewer]
        for cid in sorted_case_ids:
            if cid in outcomes:
                outcome = outcomes[cid]
            elif cid.isdigit() and int(cid) in outcomes:
                outcome = outcomes[int(cid)]
            else:
                outcome = outcomes.get(cid, "—")
            row.append(str(outcome))

        completed_str = _format_completed(item, default_total=num_cases)
        row.append(completed_str)
        case_lines.append("| " + " | ".join(row) + " |")

    # -------------------------------------------------------------------------
    # Section 2: Reviewer Metrics Summary
    # -------------------------------------------------------------------------
    summary_headers = [
        "Reviewer",
        "TP",
        "FP",
        "FN",
        "Duplicates",
        "Precision",
        "Recall",
        "F1",
        "Clean TN",
        "Clean FA",
        "Specificity",
        "Completed",
    ]
    summary_separator = ["---"] * len(summary_headers)

    summary_lines = [
        "## Reviewer Metrics Summary",
        "",
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(summary_separator) + " |",
    ]

    for item in results:
        reviewer = str(item.get("reviewer", "Unknown"))
        tp = str(item.get("tp", 0))
        fp = str(item.get("fp", 0))
        fn = str(item.get("fn", 0))
        duplicates = str(item.get("duplicates", 0))
        precision = _format_percentage(item.get("precision"))
        recall = _format_percentage(item.get("recall"))
        f1 = _format_percentage(item.get("f1"))
        clean_tn = str(item.get("clean_tn", 0))
        clean_fa = str(item.get("clean_fa", 0))
        specificity = _format_percentage(item.get("clean_specificity"))
        completed_str = _format_completed(item, default_total=num_cases)

        row = [
            reviewer,
            tp,
            fp,
            fn,
            duplicates,
            precision,
            recall,
            f1,
            clean_tn,
            clean_fa,
            specificity,
            completed_str,
        ]
        summary_lines.append("| " + " | ".join(row) + " |")

    # Combine both sections
    return "\n".join(case_lines) + "\n\n" + "\n".join(summary_lines)
