"""Verify HumanCompletenessGrader output against dataset meta scores.

Loads a dataset, checks that meta["human_completeness_score"] is populated,
runs HumanCompletenessGrader, and confirms grader output matches meta scores.
"""
import os
import argparse
import json
import logging
import asyncio
from collections import Counter

import jsonlines

from medscore_completeness import configure_logging
from medscore_completeness.utils.medical_qa_dataset import medical_qa_to_class
from medscore_completeness.graders import Grader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify HumanCompletenessGrader output for a given dataset."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write output JSONL")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MedExpert",
        choices=list(medical_qa_to_class.keys()),
        help="Dataset to verify"
    )
    parser.add_argument(
        "--dataset-kwargs",
        type=json.loads,
        default="{}",
        help="JSON string of optional arguments for the dataset class"
    )
    parser.add_argument("--n-examples", type=int, default=20, help="Number of examples to load (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def _print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


async def main():
    args = parse_args()

    configure_logging(log_level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load dataset ────────────────────────────────────────────
    _print_section(f"Loading dataset: {args.dataset}")
    dataset_kwargs = args.dataset_kwargs.copy()
    if "random_state" not in dataset_kwargs:
        dataset_kwargs["random_state"] = args.seed
    if args.n_examples != -1:
        dataset_kwargs["n_examples"] = args.n_examples

    dataset = medical_qa_to_class[args.dataset](**dataset_kwargs)
    print(f"Loaded {len(dataset)} examples.")

    # ── 2. Inspect meta["human_completeness_score"] ────────────────
    _print_section("Dataset meta scores (sample)")
    print(f"{'ID':<40} {'omission_severity':<22} {'human_score':>12}")
    print("-" * 76)
    for item in dataset:
        meta = item["meta"]
        score = meta.get("human_completeness_score")
        severity = meta.get("omission_severity", "N/A")
        print(f"{str(item['id']):<40} {str(severity):<22} {str(score):>12}")

    all_scores = [item["meta"].get("human_completeness_score") for item in dataset]
    n_none = sum(1 for s in all_scores if s is None)
    n_valid = len(all_scores) - n_none
    print(f"\nValid scores : {n_valid}/{len(all_scores)}")
    if n_none > 0:
        print(f"WARNING: {n_none} items have no human_completeness_score.")

    if n_valid > 0:
        valid_scores = [s for s in all_scores if s is not None]
        print(f"Score range  : [{min(valid_scores):.3f}, {max(valid_scores):.3f}]")
        print(f"Score mean   : {sum(valid_scores)/len(valid_scores):.3f}")
        score_counts = Counter(f"{s:.2f}" for s in valid_scores)
        print("Distribution :", dict(sorted(score_counts.items())))

    # ── 3. Run HumanCompletenessGrader ─────────────────────────────
    _print_section("Running HumanCompletenessGrader")
    grader = Grader.get_grader("completeness-human", {})
    graded = await grader(dataset)
    print(f"Graded {len(graded)} items.")

    # ── 4. Verify grader output matches meta scores ─────────────────
    _print_section("Verification: grader output vs meta score")
    mismatches = 0
    print(f"{'ID':<40} {'meta_score':>12} {'grader_score':>14} {'match':>8}")
    print("-" * 76)
    for item, result in zip(dataset, graded):
        meta_score = item["meta"].get("human_completeness_score")
        grader_score = result.get("completeness-human", {}).get("score")
        match = meta_score == grader_score
        if not match:
            mismatches += 1
        flag = "" if match else "  ← MISMATCH"
        print(f"{str(item['id']):<40} {str(meta_score):>12} {str(grader_score):>14} {str(match):>8}{flag}")

    print(f"\nMismatches: {mismatches}/{len(graded)}")
    if mismatches == 0:
        print("All grader scores match meta scores.")
    else:
        print("ERROR: Some scores do not match. Check get_human_completeness_score implementation.")

    # ── 5. Write combined output ────────────────────────────────────
    output_file = os.path.join(args.output_dir, f"{args.dataset}_completeness-human.jsonl")
    if os.path.exists(output_file) and not args.overwrite:
        logger.warning(f"Output file {output_file} already exists. Use --overwrite to replace.")
    else:
        combined = []
        for item, result in zip(dataset, graded):
            merged = result.copy()
            merged["meta"] = item.copy()
            combined.append(merged)
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(combined)
        print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
