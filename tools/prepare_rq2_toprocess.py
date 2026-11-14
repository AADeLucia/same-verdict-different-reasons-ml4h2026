"""
Normalize metric names in rq2_analysis_results.jsonl and compute the anti-join
against rq2_qualitative.jsonl to find rows not yet analyzed.

Writes:
  - rq2_qualitative_toprocess.jsonl  (rows to feed to reasoning_alignment.py / disagreement_analysis.py)
  - rq2_analysis_results.jsonl       (rewritten in-place with normalized metric names)
"""

import argparse
import collections
import json
import os
import sys


_METRIC_NAME_MAP = {
    "Analytical-Rubric":             "Omission-Harm Rubric (Zero-shot)",
    "Analytical-Rubric (FS)":        "Omission-Harm Rubric",
    "Analytical-Rubric (FSS)":       "Omission-Harm Rubric",
    "Dynamic-Checklist":             "Criteria-Checklist",
}

_DROP_METRICS = {"General-Likert", "General-Likert (FS)", "General-Likert (FSS)"}

_EXCLUDE_DATASETS = {"HealthBench", "HealthBench-MT"}
_EXCLUDE_METRICS  = {"Criteria-Checklist", "Dynamic-Checklist"}

_JOIN_KEYS = ("ID", "Metric", "Eval Model", "Dataset")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row):
    return tuple(row.get(k, "") for k in _JOIN_KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualitative", required=True,
                        help="Path to rq2_qualitative.jsonl (full, all categories)")
    parser.add_argument("--results", required=True,
                        help="Path to rq2_analysis_results.jsonl (existing LLM results)")
    parser.add_argument("--output", required=True,
                        help="Output path for rq2_qualitative_toprocess.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without writing files")
    args = parser.parse_args()

    print(f"Loading qualitative file: {args.qualitative}")
    qual_rows = load_jsonl(args.qualitative)
    print(f"  {len(qual_rows):,} rows loaded")

    # Exclude HealthBench/MT × Criteria-Checklist
    qual_filtered = [
        r for r in qual_rows
        if not (
            r.get("Dataset") in _EXCLUDE_DATASETS
            and r.get("Metric") in _EXCLUDE_METRICS
        )
    ]
    n_excluded = len(qual_rows) - len(qual_filtered)
    print(f"  {n_excluded:,} rows excluded (HealthBench/MT × Criteria-Checklist)")
    print(f"  {len(qual_filtered):,} rows remaining in scope")

    print(f"\nLoading results file: {args.results}")
    result_rows = load_jsonl(args.results)
    print(f"  {len(result_rows):,} rows loaded")

    # Normalize metric names; drop GL rows
    normalized = []
    n_dropped = 0
    for r in result_rows:
        metric = r.get("Metric", "")
        if metric in _DROP_METRICS:
            n_dropped += 1
            continue
        new_metric = _METRIC_NAME_MAP.get(metric, metric)
        r = dict(r, Metric=new_metric)
        normalized.append(r)
    n_renamed = sum(
        1 for r in normalized
        if _METRIC_NAME_MAP.get(
            [orig for orig, new in _METRIC_NAME_MAP.items() if new == r.get("Metric")][-1]
            if [orig for orig, new in _METRIC_NAME_MAP.items() if new == r.get("Metric")]
            else r.get("Metric"), r.get("Metric")
        ) != r.get("Metric")
    )
    print(f"  {n_dropped:,} GL rows dropped")
    print(f"  {len(normalized):,} rows after normalization")

    already_done = {row_key(r) for r in normalized}
    print(f"  {len(already_done):,} unique (ID, Metric, Eval Model, Dataset) keys in results")

    # Anti-join
    toprocess = [r for r in qual_filtered if row_key(r) not in already_done]
    print(f"\n{len(toprocess):,} rows to process (not yet in results)")

    # Breakdown
    counts = collections.Counter(
        (r.get("Dataset"), r.get("Metric"), r.get("analysis_category"))
        for r in toprocess
    )
    print("\nBreakdown by (Dataset, Metric, Category):")
    for (ds, m, cat), n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {ds} | {m} | {cat}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write toprocess file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_jsonl(args.output, toprocess)
    print(f"\nWrote {len(toprocess):,} rows → {args.output}")

    # Rewrite results in-place with normalized names
    write_jsonl(args.results, normalized)
    print(f"Rewrote {len(normalized):,} rows (normalized names) → {args.results}")


if __name__ == "__main__":
    main()
