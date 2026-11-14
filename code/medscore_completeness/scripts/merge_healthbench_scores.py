"""Merge two scored HealthBench JSONL files by id.

Typical use: after running compute_healthbench_human_scores.py on the
multi-turn subset, merge the result into the existing augmented_w_scores file.

    python -m medscore_completeness.scripts.merge_healthbench_scores \\
        --base   data/HealthBench/healthbench_dataset_augmented_w_scores.jsonl \\
        --new    results/healthbench_human_scores_multiturn/healthbench_human_scores_multiturn.jsonl \\
        --output data/HealthBench/healthbench_dataset_augmented_w_scores.jsonl
"""
import argparse
from collections import OrderedDict

import jsonlines


def main():
    parser = argparse.ArgumentParser(
        description="Merge two scored HealthBench JSONL files by id."
    )
    parser.add_argument(
        "--base", required=True,
        help="Existing scored JSONL (rows preserved as-is).",
    )
    parser.add_argument(
        "--new", required=True,
        help="New scored JSONL to merge in. Duplicate ids are skipped.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: overwrite --base).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary without writing output.",
    )
    args = parser.parse_args()
    output = args.output or args.base

    merged: OrderedDict = OrderedDict()
    with jsonlines.open(args.base) as reader:
        for row in reader:
            merged[row["id"]] = row
    n_base = len(merged)

    n_added = n_skipped = 0
    with jsonlines.open(args.new) as reader:
        for row in reader:
            if row["id"] in merged:
                n_skipped += 1
            else:
                merged[row["id"]] = row
                n_added += 1

    print(f"Base rows:          {n_base}")
    print(f"New rows added:     {n_added}")
    print(f"Duplicates skipped: {n_skipped}")
    print(f"Total rows:         {len(merged)}")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    with jsonlines.open(output, "w") as writer:
        writer.write_all(merged.values())
    print(f"Written → {output}")


if __name__ == "__main__":
    main()
