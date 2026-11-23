"""Review Evaluator for the Proofline AI Code Review Benchmark.

This module evaluates AI code reviewers against known ground-truth defects
across benchmark cases, calculating:
- Bug-level metrics: True Positives (TP), False Positives (FP), False Negatives (FN),
  Duplicates, Precision, Recall, and F1 score.
- Clean-case metrics: True Negatives (TN), False Alarms (FA), and Clean Specificity.
- Case-level composite outcomes (e.g. TP, FP, FN, TN, TP+FP, TP+FN, FN+FP, TP+FN+FP,
  QUOTA, ERROR, —).
- Operational status handling (completed, quota_exhausted, error, unattempted).
"""

import json
from pathlib import Path
from typing import Any


def calculate_metrics(
    tp: int, fp: int, fn: int
) -> tuple[float | None, float | None, float | None]:
    """Calculate precision, recall, and F1 score with division-by-zero protection.

    Rules:
    - If there are no reported defect findings (TP + FP == 0), Precision is None.
    - If there are no ground-truth bugs (TP + FN == 0), Recall is None.
    - If Precision or Recall is None, F1 score is None.
    - If Precision + Recall == 0, F1 score is 0.0.

    Args:
        tp: Count of true positive defect findings.
        fp: Count of false positive defect findings.
        fn: Count of false negative (missed) defects.

    Returns:
        Tuple of (precision, recall, f1).
    """
    reported = tp + fp
    precision = (tp / reported) if reported > 0 else None

    ground_truth = tp + fn
    recall = (tp / ground_truth) if ground_truth > 0 else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_specificity(clean_tn: int, clean_fa: int) -> float | None:
    """Calculate clean-case specificity with division-by-zero protection.

    Clean-case specificity measures how reliably a reviewer avoids false alarms
    on defect-free (control) cases.

    Formula:
        TN / (TN + False Alarms)

    Args:
        clean_tn: Count of clean cases with zero false alarms.
        clean_fa: Count of clean cases with at least one false positive.

    Returns:
        Specificity ratio (0.0 to 1.0), or None if no clean cases were evaluated.
    """
    total_clean = clean_tn + clean_fa
    return (clean_tn / total_clean) if total_clean > 0 else None


def _safe_list(val: Any) -> list[Any]:
    """Return val if it is a list, otherwise an empty list.

    Safely normalizes optional or nullable collection fields (e.g. JSON null
    or missing fields) to prevent TypeErrors during iteration.
    """
    return val if isinstance(val, list) else []


def _normalize_status(status: Any) -> str:
    """Normalize a review status string by stripping whitespace and converting to lowercase."""
    if not isinstance(status, str):
        return "unattempted"
    normalized = status.strip().lower()
    return normalized if normalized else "unattempted"


def _fp_finding_fingerprint(finding: dict[str, Any], index: int) -> tuple[Any, ...]:
    """Generate a fingerprint for a false-positive finding within a case.

    Rules:
    - When both matched_bug_id and description exist, fingerprint includes both
      so different descriptions sharing the same fake ID are not merged.
    - When matched_bug_id is missing, fingerprint uses the normalized description.
    - When only matched_bug_id exists, fingerprint uses the normalized ID.
    - When neither usable ID nor description exists, returns a unique index-based
      fingerprint so unrelated findings are not accidentally merged.
    """
    raw_desc = finding.get("description")
    norm_desc = raw_desc.strip() if isinstance(raw_desc, str) and raw_desc.strip() else None

    raw_id = finding.get("matched_bug_id")
    norm_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else None

    if norm_id and norm_desc:
        return ("id_desc", norm_id, norm_desc)
    elif norm_desc:
        return ("desc", norm_desc)
    elif norm_id:
        return ("id", norm_id)
    return ("unique", index)


def _load_json_data(data_input: str | Path | dict[str, Any] | list[Any]) -> Any:
    """Helper to load a JSON dictionary or list from a file path or dict/list."""
    if isinstance(data_input, (str, Path)):
        path = Path(data_input)
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, (dict, list)):
                return loaded
            return {}
    elif isinstance(data_input, (dict, list)):
        return data_input
    return {}


def _extract_reviewer_cases(
    data: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract reviewer submissions into a normalized mapping:

    { reviewer_name: { case_id: { "status": str, "findings": list } } }
    """
    reviewer_cases: dict[str, dict[str, dict[str, Any]]] = {}

    if isinstance(data, dict):
        # 1. Top-level "reviews" list (the approved separate reviews format)
        if "reviews" in data and isinstance(data["reviews"], list):
            for review in data["reviews"]:
                if not isinstance(review, dict):
                    continue
                reviewer_name = review.get("reviewer", "Unknown")
                case_id = review.get("case_id")
                if not case_id:
                    continue
                if reviewer_name not in reviewer_cases:
                    reviewer_cases[reviewer_name] = {}
                reviewer_cases[reviewer_name][case_id] = {
                    "status": _normalize_status(review.get("status")),
                    "findings": _safe_list(review.get("findings")),
                }

        # 2. Top-level "cases" list (combined dataset or case-grouped reviews)
        if "cases" in data and isinstance(data["cases"], list):
            for case in data["cases"]:
                if not isinstance(case, dict):
                    continue
                case_id = case.get("case_id")
                if not case_id:
                    continue
                # Support direct review records inside cases list
                if "reviewer" in case:
                    reviewer_name = case.get("reviewer", "Unknown")
                    if reviewer_name not in reviewer_cases:
                        reviewer_cases[reviewer_name] = {}
                    reviewer_cases[reviewer_name][case_id] = {
                        "status": _normalize_status(case.get("status")),
                        "findings": _safe_list(case.get("findings")),
                    }
                    continue
                for rev in _safe_list(case.get("reviewers")):
                    if not isinstance(rev, dict):
                        continue
                    reviewer_name = rev.get("reviewer", "Unknown")
                    if reviewer_name not in reviewer_cases:
                        reviewer_cases[reviewer_name] = {}
                    reviewer_cases[reviewer_name][case_id] = {
                        "status": _normalize_status(rev.get("status")),
                        "findings": _safe_list(rev.get("findings")),
                    }

        # 3. Top-level "reviewers" list (reviewer-grouped format)
        if "reviewers" in data and isinstance(data["reviewers"], list):
            for rev in data["reviewers"]:
                if not isinstance(rev, dict):
                    continue
                reviewer_name = rev.get("reviewer", "Unknown")
                if reviewer_name not in reviewer_cases:
                    reviewer_cases[reviewer_name] = {}
                entries = _safe_list(rev.get("reviews")) or _safe_list(rev.get("cases"))
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    case_id = entry.get("case_id")
                    if not case_id:
                        continue
                    reviewer_cases[reviewer_name][case_id] = {
                        "status": _normalize_status(entry.get("status")),
                        "findings": _safe_list(entry.get("findings")),
                    }

    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "reviewer" in item and "case_id" in item:
                reviewer_name = item.get("reviewer", "Unknown")
                case_id = item.get("case_id")
                if not case_id:
                    continue
                if reviewer_name not in reviewer_cases:
                    reviewer_cases[reviewer_name] = {}
                reviewer_cases[reviewer_name][case_id] = {
                    "status": _normalize_status(item.get("status")),
                    "findings": _safe_list(item.get("findings")),
                }
            elif "case_id" in item and "reviewers" in item:
                case_id = item.get("case_id")
                if not case_id:
                    continue
                for rev in _safe_list(item.get("reviewers")):
                    if not isinstance(rev, dict):
                        continue
                    reviewer_name = rev.get("reviewer", "Unknown")
                    if reviewer_name not in reviewer_cases:
                        reviewer_cases[reviewer_name] = {}
                    reviewer_cases[reviewer_name][case_id] = {
                        "status": _normalize_status(rev.get("status")),
                        "findings": _safe_list(rev.get("findings")),
                    }

    return reviewer_cases


def evaluate_benchmark(
    ground_truth_input: str | Path | dict[str, Any],
    reviews_input: str | Path | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate reviewer performance against ground-truth benchmark cases.

    Supports both:
    - Combined dataset input (ground_truth_input contains cases with both
      ground_truth and reviewers).
    - Separate inputs (ground_truth_input provides cases with ground_truth;
      reviews_input provides reviewer submissions).

    Args:
        ground_truth_input: Path or dict containing ground-truth cases.
        reviews_input: Optional path or dict containing reviewer results.

    Returns:
        List of dictionaries with evaluation metrics per reviewer.
    """
    gt_data = _load_json_data(ground_truth_input)

    # If separate reviews_input is provided, load it; otherwise extract from combined gt_data
    if reviews_input is not None:
        rev_data = _load_json_data(reviews_input)
    else:
        rev_data = gt_data

    # Map ground truth bugs per case
    gt_by_case: dict[str, set[str]] = {}
    all_case_ids: list[str] = []

    if isinstance(gt_data, list):
        gt_cases = gt_data
    elif isinstance(gt_data, dict):
        gt_cases = _safe_list(gt_data.get("cases"))
    else:
        gt_cases = []

    for case in gt_cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        if not case_id or case_id in gt_by_case:
            continue
        all_case_ids.append(case_id)
        gt_list = _safe_list(case.get("ground_truth"))
        gt_by_case[case_id] = {
            b["bug_id"] for b in gt_list if isinstance(b, dict) and "bug_id" in b
        }

    # Extract reviewer cases
    reviewer_cases = _extract_reviewer_cases(rev_data)

    # If ground truth had no cases, fall back to case IDs from reviews
    if not all_case_ids:
        seen: set[str] = set()
        for rev_map in reviewer_cases.values():
            for cid in rev_map:
                if cid not in seen:
                    seen.add(cid)
                    all_case_ids.append(cid)

    gt_case_ids_set = set(all_case_ids)

    results: list[dict[str, Any]] = []

    for reviewer_name, cases_map in reviewer_cases.items():
        unknown_cases = sorted(
            [cid for cid in cases_map if cid not in gt_case_ids_set and str(cid) not in gt_case_ids_set],
            key=str,
        )
        tp = 0
        fp = 0
        fn = 0
        duplicates = 0
        clean_tn = 0
        clean_fa = 0
        completed_cases = 0
        case_outcomes: dict[str, str] = {}

        for case_id in all_case_ids:
            gt_bug_ids = gt_by_case.get(case_id, set())
            is_clean_case = len(gt_bug_ids) == 0

            rev_case = cases_map.get(case_id)
            if rev_case is None:
                case_outcomes[case_id] = "—"
                continue

            status = _normalize_status(rev_case.get("status"))
            if status in ("quota_exhausted", "quota"):
                case_outcomes[case_id] = "QUOTA"
                continue
            elif status == "error":
                case_outcomes[case_id] = "ERROR"
                continue
            elif status == "unattempted":
                case_outcomes[case_id] = "—"
                continue
            elif status != "completed":
                case_outcomes[case_id] = "—"
                continue

            # Reviewer completed this case
            completed_cases += 1

            matched_in_case: set[str] = set()
            seen_fp_in_case: set[tuple[Any, ...]] = set()
            case_fp = 0
            case_duplicates = 0

            for idx, finding in enumerate(_safe_list(rev_case.get("findings"))):
                if not isinstance(finding, dict):
                    continue
                # Ignore non-defect findings (only type == 'bug' counts; default to 'bug' for backward compatibility)
                finding_type = finding.get("type", "bug")
                if finding_type != "bug":
                    continue

                matched_id = finding.get("matched_bug_id")

                if matched_id and matched_id in gt_bug_ids:
                    if matched_id not in matched_in_case:
                        matched_in_case.add(matched_id)
                    else:
                        # Repeated match within the same case is a duplicate
                        case_duplicates += 1
                else:
                    # Defect claimed but not in ground truth for this case
                    fp_key = _fp_finding_fingerprint(finding, idx)
                    if fp_key not in seen_fp_in_case:
                        seen_fp_in_case.add(fp_key)
                        case_fp += 1
                    else:
                        # Repeated false-positive claim within the same case is a duplicate
                        case_duplicates += 1

            case_tp = len(matched_in_case)
            case_fn = len(gt_bug_ids) - case_tp

            tp += case_tp
            fp += case_fp
            fn += case_fn
            duplicates += case_duplicates

            # Determine case outcome and clean-case stats
            if is_clean_case:
                if case_fp == 0:
                    clean_tn += 1
                    case_outcomes[case_id] = "TN"
                else:
                    clean_fa += 1
                    case_outcomes[case_id] = "FP"
            else:
                parts = []
                if case_tp > 0:
                    parts.append("TP")
                if case_fn > 0:
                    parts.append("FN")
                if case_fp > 0:
                    parts.append("FP")
                case_outcomes[case_id] = "+".join(parts) if parts else "FN"

        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        specificity = calculate_specificity(clean_tn, clean_fa)

        results.append(
            {
                "reviewer": reviewer_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cases_evaluated": completed_cases,
                "duplicates": duplicates,
                "clean_tn": clean_tn,
                "clean_fa": clean_fa,
                "clean_specificity": specificity,
                "total_cases": len(all_case_ids),
                "completed_cases": completed_cases,
                "case_outcomes": case_outcomes,
                "unknown_cases": unknown_cases,
            }
        )

    return results


def evaluate_dataset(
    file_path: str | Path,
    reviews_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load a benchmark review dataset from a JSON file and evaluate completed reviewers.

    Backward-compatible entry point for existing test suites and CLI callers.

    Args:
        file_path: Path to the JSON dataset file (combined format or ground truth).
        reviews_path: Optional path to separate reviewer findings file.

    Returns:
        List of dictionaries containing evaluation metrics per reviewer.
    """
    return evaluate_benchmark(file_path, reviews_input=reviews_path)


if __name__ == "__main__":
    import pprint

    benchmark_dir = Path(__file__).parent
    sample_file = benchmark_dir / "sample_input.json"
    gt_file = benchmark_dir / "ground_truth.json"

    if sample_file.exists():
        print("Evaluating sample_input.json (combined format):")
        sample_results = evaluate_dataset(sample_file)
        pprint.pprint(sample_results)
        print()

    if gt_file.exists():
        print(f"Ground truth dataset loaded from {gt_file.name}:")
        gt_data = _load_json_data(gt_file)
        print(f"Total benchmark cases: {len(gt_data.get('cases', []))}")
