# AI Review Benchmarks & Evaluator

A benchmark evaluator and comparison matrix generator for measuring automated AI pull request reviewers against ground-truth defects across standardized benchmark pull requests.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements & Installation](#requirements--installation)
- [CLI Usage](#cli-usage)
  - [Positional Dataset Mode](#positional-dataset-mode)
  - [Separate-File Mode](#separate-file-mode)
  - [CLI Exit Codes](#cli-exit-codes)
- [Input JSON Schemas](#input-json-schemas)
  - [1. Ground-Truth Schema](#1-ground-truth-schema)
  - [2. Separate Reviews Schema](#2-separate-reviews-schema)
  - [3. Combined Positional Schema](#3-combined-positional-schema)
- [Metric Definitions](#metric-definitions)
  - [Classification Metrics (Bug Cases)](#classification-metrics-bug-cases)
  - [Control Metrics (Clean Cases)](#control-metrics-clean-cases)
  - [Operational & Completion Metrics](#operational--completion-metrics)
- [Case Outcome Codes](#case-outcome-codes)
- [Duplicate Findings & Unknown Case Handling](#duplicate-findings--unknown-case-handling)
- [Canonical Benchmark Example Output](#canonical-benchmark-example-output)
- [Running the Test Suite](#running-the-test-suite)
- [Original Benchmark Challenge Reference](#original-benchmark-challenge-reference)

---

## Project Overview

The **Proofline Review Evaluator** evaluates how accurately automated AI code review tools detect defects, avoid hallucinations, and handle clean code changes across benchmark pull requests.

Given ground-truth defect specifications and reviewer submissions, the evaluator:
1. Matches reviewer findings against known bugs on a per-case basis.
2. Deduplicates multiple reports of the same defect or repeated false-positive claims.
3. Calculates standard classification metrics (**Precision**, **Recall**, **F1**) as well as clean-case control metrics (**Clean Specificity**).
4. Generates an executive Markdown report containing:
   - **Reviewer-by-Case Matrix**: Dynamic case-level outcomes showing completed, missed, mixed, quota-stopped, or unattempted cases.
   - **Reviewer Metrics Summary**: Aggregated performance statistics across all evaluated reviewers.

---

## Features

- **Classification & Control Metrics**: True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, F1 score, Clean True Negatives (TN), Clean False Alarms (FA), and Clean Specificity.
- **Smart Duplicate Detection**:
  - Repeated matches of a ground-truth bug within a case increment the `Duplicates` counter and do not artificially inflate TP.
  - Repeated false-positive claims within a case (fingerprinted by bug ID and normalized description) increment `Duplicates` and do not artificially inflate FP.
  - Non-defect comments (e.g. style, documentation, suggestions) are ignored from defect scoring.
- **Operational Review Statuses**: Accurately accounts for incomplete runs (`quota` / `quota_exhausted`, `error`, `unattempted`) without penalizing accuracy denominators.
- **Unknown Case Isolation**: Detects cases submitted by reviewers that are not part of the ground truth, emits a warning to `stderr`, and prevents benchmark corruption.
- **Dual Section Markdown Output**: Outputs clean, publication-ready Markdown tables.
- **Flexible JSON Formats**: Supports separate files, combined datasets, reviewer-grouped formats, and raw list arrays.
- **Zero External Dependencies**: Implemented purely using the Python standard library (`json`, `argparse`, `pathlib`, `re`, `sys`).

---

## Requirements & Installation

- **Python Version**: Python 3.10 or later (tested on 3.10, 3.11, 3.12, and 3.13).
- **Dependencies**: None. Only the Python Standard Library is required.

Clone the repository and run all commands directly from the repository root:

```bash
git clone https://github.com/litheshpa80/ai-review-benchmarks-lithesh.git
cd ai-review-benchmarks-lithesh
```

---

## CLI Usage

The evaluator is executed via the Python module entrypoint `benchmark.cli`.

### Positional Dataset Mode
For combined datasets where cases contain both ground-truth definitions and reviewer findings:

```bash
python -m benchmark.cli benchmark/sample_input.json
```

### Separate-File Mode
For benchmarks where ground-truth definitions and reviewer submissions are maintained in separate JSON files:

```bash
python -m benchmark.cli --ground-truth benchmark/ground_truth.json --reviews benchmark/reviews.json
```

Short flags `-g` and `-r` are also supported:

```bash
python -m benchmark.cli -g benchmark/ground_truth.json -r benchmark/reviews.json
```

### CLI Exit Codes

| Exit Code | Meaning | Description |
| :---: | :--- | :--- |
| **`0`** | **Success** | Benchmark evaluated successfully; Markdown matrix printed to `stdout`. |
| **`1`** | **Error** | File not found, path is a directory, invalid JSON (with failing path identified), conflicting arguments, or empty evaluation results. Error message printed to `stderr`. |
| **`2`** | **Usage Error** | Missing required arguments (e.g., neither positional path nor flags provided, or `-h` / `--help`). |

---

## Input JSON Schemas

### 1. Ground-Truth Schema
Used in separate-file mode (`--ground-truth <path>`). Contains an array of benchmark cases with known bugs:

```json
{
  "cases": [
    {
      "case_id": "case-01",
      "title": "Page report history by scheduled time",
      "ground_truth": [
        {
          "bug_id": "BUG-01",
          "category": "pagination",
          "description": "Keyset pagination cursor uses only scheduled_at without an id tie-breaker.",
          "severity": "high"
        }
      ]
    },
    {
      "case_id": "case-07",
      "title": "Index pending deliveries",
      "ground_truth": []
    }
  ]
}
```
*Note: A case with `ground_truth: []` represents a clean control case with no defects.*

### 2. Separate Reviews Schema
Used in separate-file mode (`--reviews <path>`). Contains reviewer review submissions per case:

```json
{
  "reviews": [
    {
      "reviewer": "Reviewer-A",
      "case_id": "case-01",
      "status": "completed",
      "findings": [
        {
          "matched_bug_id": "BUG-01",
          "type": "bug",
          "description": "Pagination cursor lacks unique tie-breaker."
        }
      ]
    },
    {
      "reviewer": "Reviewer-B",
      "case_id": "case-07",
      "status": "completed",
      "findings": [
        {
          "matched_bug_id": null,
          "type": "bug",
          "description": "Hallucinated defect on clean code."
        }
      ]
    },
    {
      "reviewer": "Reviewer-D",
      "case_id": "case-01",
      "status": "quota",
      "findings": []
    }
  ]
}
```

### 3. Combined Positional Schema
Used in positional mode (`python -m benchmark.cli <path>`). Embeds reviewers directly inside cases:

```json
{
  "cases": [
    {
      "case_id": "case-01",
      "ground_truth": [
        {
          "bug_id": "BUG-01",
          "category": "pagination",
          "description": "Cursor tie-breaker missing."
        }
      ],
      "reviewers": [
        {
          "reviewer": "Reviewer-A",
          "status": "completed",
          "findings": [
            {
              "matched_bug_id": "BUG-01",
              "type": "bug"
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Metric Definitions

### Classification Metrics (Bug Cases)

- **True Positives (TP)**: Count of ground-truth bugs correctly identified by the reviewer (`type: "bug"` and matching a valid `bug_id`).
- **False Positives (FP)**: Count of reported findings (`type: "bug"`) that do not match any ground-truth bug for that case.
- **False Negatives (FN)**: Count of ground-truth bugs that the reviewer failed to detect.
- **Duplicates**: Count of repeated findings within the same case:
  - If a reviewer reports the same ground-truth bug multiple times in a case, subsequent reports count as duplicates instead of increasing TP.
  - If a reviewer reports the same false-positive claim multiple times in a case, subsequent reports count as duplicates instead of increasing FP.
- **Precision**: Ratio of valid bugs to all claimed defects:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  *Displays as `N/A` if no defect findings were reported ($TP + FP = 0$).*
- **Recall**: Ratio of detected bugs to total ground-truth bugs:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  *Displays as `N/A` if no ground-truth bugs exist ($TP + FN = 0$).*
- **F1 Score**: Harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  *Displays as `0.0%` if $\text{Precision} + \text{Recall} = 0$, or `N/A` if Precision or Recall is `N/A`.*

### Control Metrics (Clean Cases)

A **clean case** is a control pull request containing zero ground-truth bugs (e.g. `case-07`).

- **Clean TN (True Negative)**: Clean cases reviewed with zero false alarms ($FP = 0$).
- **Clean FA (False Alarm)**: Clean cases where the reviewer falsely reported one or more defects ($FP > 0$).
- **Clean Specificity**: The reviewer's ability to remain quiet on clean changes:
  $$\text{Clean Specificity} = \frac{\text{Clean TN}}{\text{Clean TN} + \text{Clean FA}}$$
  *Displays as `N/A` if no clean cases were completed.*

### Operational & Completion Metrics

- **Completed**: Ratio of completed cases to total cases, formatted as `completed/total` (e.g., `8/8` or `1/8`).
- **Incomplete Review Handling**: Reviews with status `unattempted`, `quota` / `quota_exhausted`, or `error` do **not** increment completed cases and do not count toward TP, FP, or FN.

---

## Case Outcome Codes

In the **Reviewer-by-Case Matrix**, each cell displays the exact outcome for that reviewer and case:

### For Defect Cases (Cases with $\ge 1$ Bug)
| Code | Meaning | Details |
| :--- | :--- | :--- |
| **`TP`** | Full Detection | Reviewer found all bugs; zero false positives and zero misses. |
| **`FN`** | Complete Miss | Reviewer missed all bugs; zero true positives and zero false positives. |
| **`FP`** | Pure Hallucination | Reviewer found zero bugs and reported one or more false defects. |
| **`TP+FN`** | Partial Detection | Reviewer detected some bugs but missed others; zero false positives. |
| **`TP+FP`** | Detection with Noise | Reviewer found all bugs, but also reported one or more false positives. |
| **`FN+FP`** | Miss with Noise | Reviewer missed all bugs, and reported one or more false positives. |
| **`TP+FN+FP`** | Partial with Noise | Reviewer found some bugs, missed others, and reported false positives. |

### For Clean Control Cases (Cases with 0 Bugs)
| Code | Meaning | Details |
| :--- | :--- | :--- |
| **`TN`** | True Negative | Reviewer correctly raised zero defect findings on clean code. |
| **`FP`** | False Alarm | Reviewer hallucinated one or more defects on clean code. |

### Operational / Incomplete Reviews
| Code | Meaning | Details |
| :--- | :--- | :--- |
| **`QUOTA`** | Quota Exhausted | Review aborted due to provider quota or rate limiting (`status: "quota"` or `"quota_exhausted"`). |
| **`ERROR`** | Execution Error | Review failed due to provider or tool execution error (`status: "error"`). |
| **`—`** | Unattempted | Case was not attempted or was omitted by the reviewer (`status: "unattempted"`). |

---

## Duplicate Findings & Unknown Case Handling

### Duplicate Findings
- **Defect Findings (`type: "bug"`)**: Only findings with `type: "bug"` (or untyped for backward compatibility) count toward defect evaluation. Non-defect comments (`type: "style"`, `"doc"`, `"nit"`) are ignored.
- **True Positive Deduplication**: When multiple findings match the same `matched_bug_id` in a case, the first match earns $TP = 1$, and all subsequent matches increment `Duplicates += 1`.
- **False Positive Deduplication**: When multiple findings report false defects in a case, each unique claim earns $FP = 1$. Identical duplicate claims (fingerprinted by `matched_bug_id` and normalized description whitespace) increment `Duplicates += 1` instead of counting as additional false positives.
- **Cross-Case Independence**: Duplicate tracking is scoped per case. A recurring finding reported across two separate cases counts independently in both cases.

### Unknown Case ID Warnings
If a reviewer submits findings for a case ID that does not exist in the ground truth (e.g., `case-99`):
1. A warning is emitted to `stderr`:
   ```
   Warning: Reviewer 'Reviewer-A' submitted unknown case IDs not present in ground truth: ['case-99']
   ```
2. The unknown cases are stored in the reviewer's result under `unknown_cases`.
3. Unknown cases **do not affect** the reviewer's TP, FP, FN, precision, recall, F1, clean specificity, completed cases, or total cases.
4. Matrix columns display only valid ground-truth cases. The CLI exits successfully with code `0`.

---

## Canonical Benchmark Example Output

Evaluating the included benchmark datasets:

```bash
python -m benchmark.cli --ground-truth benchmark/ground_truth.json --reviews benchmark/reviews.json
```

Outputs the following publication-ready Markdown tables:

## Reviewer-by-Case Matrix

| Reviewer | case-01 | case-02 | case-03 | case-04 | case-05 | case-06 | case-07 | case-08 | Completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reviewer-A | TP | TP | TP | TP | TP | TP | TN | TP | 8/8 |
| Reviewer-B | TP | FN | TP | FN | TP | TP | FP | FN | 8/8 |
| Reviewer-C | TP | TP | TP | TP | TP | TP | TN | TP | 8/8 |
| Reviewer-D | QUOTA | ERROR | — | — | — | — | — | — | 0/8 |
| Reviewer-E | — | — | — | — | — | — | FP | — | 1/8 |

## Reviewer Metrics Summary

| Reviewer | TP | FP | FN | Duplicates | Precision | Recall | F1 | Clean TN | Clean FA | Specificity | Completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reviewer-A | 7 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1 | 0 | 100.0% | 8/8 |
| Reviewer-B | 4 | 1 | 3 | 0 | 80.0% | 57.1% | 66.7% | 0 | 1 | 0.0% | 8/8 |
| Reviewer-C | 7 | 0 | 0 | 1 | 100.0% | 100.0% | 100.0% | 1 | 0 | 100.0% | 8/8 |
| Reviewer-D | 0 | 0 | 0 | 0 | N/A | N/A | N/A | 0 | 0 | N/A | 0/8 |
| Reviewer-E | 0 | 1 | 0 | 0 | 0.0% | N/A | N/A | 0 | 1 | 0.0% | 1/8 |

---

## Running the Test Suite

The test suite contains 88 comprehensive unit and regression tests:

```bash
python -m unittest discover -s tests -v
```

All tests execute with standard library modules and take under 0.1 seconds to run.

---

## Original Benchmark Challenge Reference

The benchmark evaluates changes against a Python scheduled reporting service:

- **Sample Application**: Scheduled report delivery service using SQLite 3.35+. See [Application Guide](docs/report-service.md).
- **Eight Review Cases**: Located in `cases/01` through `cases/08`. Each directory contains proposed changes, reference patches, and deployment snapshots. See [Review Cases Guide](cases/README.md).
- **Experiment Protocol**: Guidelines for isolating answer keys, controlling deployment context, and avoiding data leakage. See [Experiment Guide](docs/experiment.md).
- **AWS Deployment**: Deploying and verifying report service workloads via CloudFormation and AWS Lambda. See [AWS Deployment Guide](docs/aws.md) and [GitHub Actions Setup](docs/github-aws.md).
