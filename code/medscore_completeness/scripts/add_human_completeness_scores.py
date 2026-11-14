"""Add human completeness scores to existing grading result directories.

For each results directory containing a `{dataset}_combined.jsonl`, this script:
  1. Loads the dataset class and runs HumanCompletenessGrader to build an
     id → completeness-human payload mapping.
  2. Writes `{dataset}_completeness-human.jsonl` alongside the other metric files.
  3. Patches `{dataset}_combined.jsonl` in-place: adds the `completeness-human` key
     and updates `meta.meta.human_completeness_score` for downstream analysis.

Usage:
    python -m medscore_completeness.scripts.add_human_completeness_scores \
        --results-dir results/results_jan_arr_rebuttal \
        [--datasets HealthBench MedExpert] \
        [--overwrite]
"""
import argparse
import asyncio
import logging
import os

import jsonlines

from medscore_completeness import configure_logging
from medscore_completeness.graders.HumanCompletenessGrader import HumanCompletenessGrader
from medscore_completeness.utils.medical_qa_dataset import medical_qa_to_class

configure_logging()
logger = logging.getLogger(__name__)

HUMAN_METRIC_KEY = "completeness-human"
SCORING_MODEL = "Human"

HEALTHBENCH_DATASETS = {"HealthBench"}

_grader = HumanCompletenessGrader()


def build_score_map(dataset_name: str) -> dict:
    """Return {id: completeness-human payload} for the given dataset.

    Values are produced by HumanCompletenessGrader so the format is identical
    to what the grader writes during a normal grading run.

    HealthBench is read directly from the augmented JSONL to avoid the raw-format
    parsing step required by HealthBenchDataset._load_data.
    All other datasets are loaded via their dataset class.
    """
    if dataset_name in HEALTHBENCH_DATASETS:
        return _build_healthbench_score_map()

    dataset_class = medical_qa_to_class[dataset_name]
    dataset = dataset_class()
    async def _run():
        return await asyncio.gather(*[_grader._individual_call(item) for item in dataset])
    results = asyncio.run(_run())
    score_map = {r["id"]: r[HUMAN_METRIC_KEY] for r in results}
    logger.info(f"Loaded {len(score_map)} scores for {dataset_name}")
    return score_map


def _build_healthbench_score_map() -> dict:
    """Read human scores directly from the pre-computed augmented HealthBench JSONL."""
    path = os.path.join(os.environ["PROJECT_DIR"], "data/HealthBench/healthbench_dataset_augmented_w_scores.jsonl")
    score_map = {}
    with jsonlines.open(path) as reader:
        for item in reader:
            meta = item.get("meta", {})
            score = meta.get("human_completeness_score_unweighted", meta.get("human_completeness_score"))
            score_map[item["id"]] = {"scoring_model": SCORING_MODEL, "score": score, "points": None, "total_points": None}
    logger.info(f"Loaded {len(score_map)} HealthBench scores from {path}")
    return score_map


def find_combined_files(results_dir: str, dataset_filter: list) -> list:
    """Walk results_dir and return paths to all matching *_combined.jsonl files."""
    paths = []
    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith("_combined.jsonl"):
                continue
            dataset_name = fname.replace("_combined.jsonl", "")
            if dataset_filter and dataset_name not in dataset_filter:
                continue
            paths.append((os.path.join(root, fname), dataset_name))
    return paths


def process_combined_file(combined_path: str, dataset_name: str, score_map: dict, overwrite: bool):
    human_path = combined_path.replace("_combined.jsonl", f"_{HUMAN_METRIC_KEY}.jsonl")

    if os.path.exists(human_path) and not overwrite:
        logger.info(f"Skipping {combined_path} — {human_path} already exists (use --overwrite to rerun)")
        return

    # Read existing combined
    with jsonlines.open(combined_path) as reader:
        items = list(reader)

    human_records = []
    n_missing = 0
    for item in items:
        item_id = item["id"]
        human_metric = score_map.get(item_id)
        if human_metric is None or human_metric.get("score") is None:
            logger.warning(f"No human score found for id '{item_id}' in {dataset_name}")
            n_missing += 1
            human_metric = {"scoring_model": SCORING_MODEL, "score": None, "points": None, "total_points": None}

        human_record = {"id": item_id, HUMAN_METRIC_KEY: human_metric}
        human_records.append(human_record)

        # Patch combined item in-place
        item[HUMAN_METRIC_KEY] = human_metric
        if "meta" in item and "meta" in item["meta"]:
            item["meta"]["meta"]["human_completeness_score"] = human_metric.get("score")

    if n_missing:
        logger.warning(f"{n_missing}/{len(items)} items had no human score in {combined_path}")

    # Write completeness-human metric file
    with jsonlines.open(human_path, "w") as writer:
        writer.write_all(human_records)
    logger.info(f"Wrote {human_path}")

    # Write patched combined
    with jsonlines.open(combined_path, "w") as writer:
        writer.write_all(items)
    logger.info(f"Updated {combined_path}")


###############
# Main
###############
def parse_args():
    parser = argparse.ArgumentParser(description="Add human completeness scores to existing result directories.")
    parser.add_argument("--results-dir", nargs="+", required=True, help="Root result directories to search.")
    parser.add_argument("--datasets", nargs="*", default=[], choices=list(medical_qa_to_class.keys()), help="Restrict to these dataset names. Defaults to all found.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completeness-human files.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect all combined files across all results dirs
    all_files = []
    for results_dir in args.results_dir:
        found = find_combined_files(results_dir, args.datasets)
        logger.info(f"Found {len(found)} combined files in {results_dir}")
        all_files.extend(found)

    if not all_files:
        logger.warning("No combined files found. Check --results-dir and --datasets.")
        return

    # Build score maps once per dataset (they can be large)
    dataset_names = {ds for _, ds in all_files}
    score_maps = {}
    for dataset_name in dataset_names:
        if dataset_name not in medical_qa_to_class:
            logger.warning(f"Unknown dataset '{dataset_name}', skipping.")
            continue
        logger.info(f"Loading dataset {dataset_name}...")
        score_maps[dataset_name] = build_score_map(dataset_name)

    # Process each file
    for combined_path, dataset_name in all_files:
        if dataset_name not in score_maps:
            continue
        logger.info(f"Processing {combined_path}")
        process_combined_file(combined_path, dataset_name, score_maps[dataset_name], args.overwrite)

    logger.info("Done.")


if __name__ == "__main__":
    main()
