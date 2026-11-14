"""Grade Responses to Medical Questions

Modeled after OpenAI simple-evals healthbench_eval.py
- https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
"""
import os
import argparse
import logging
import json
from typing import Any, Dict, List, Optional, override
import logging
import asyncio
import time
from datetime import timedelta

import jsonlines
from openai.types.chat.chat_completion import ChatCompletion

from medscore_completeness.utils import redact_secrets
from medscore_completeness import configure_logging
from medscore_completeness.utils.medical_qa_dataset import medical_qa_to_class
from medscore_completeness.utils.configure_graders import load_config
from medscore_completeness.graders import (
    Grader
)

AVAILABLE_GRADERS = Grader.get_available_graders()

############
# Main
############
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade responses using various graders.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files instead of skipping them.")
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="MedExpert", choices=medical_qa_to_class.keys())
    parser.add_argument("--dataset-kwargs", type=json.loads, default="{}", help="JSON string of optional arguments for the dataset class (e.g. '{\"include_multiturn\": true}')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Grader settings
    parser.add_argument("--config", type=str, default=f"{os.environ['PROJECT_DIR']}/config.yaml", help="Path to grader config file (YAML)")
    parser.add_argument("--metrics", nargs="+", default=AVAILABLE_GRADERS, choices=AVAILABLE_GRADERS, help="Metrics to use")
    args = parser.parse_args()
    return args


async def main():
    # Parse arguments
    args = parse_args()

    # Configure logging
    configure_logging(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        log_file=os.path.join(args.output_dir, "grading.log"),
        filemode="w" if args.overwrite else "a"
    )
    logger = logging.getLogger(__name__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # load grader config
    grader_configs = load_config(args.config)

    # Save args
    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w") as f:
        args_dump = {**vars(args), **grader_configs}
        json.dump(redact_secrets(args_dump), f, indent=4)

    # Load dataset for inference
    dataset_class = medical_qa_to_class[args.dataset]
    if "seed" not in args.dataset_kwargs:
        args.dataset_kwargs["random_state"] = args.seed
    logger.info(f"Loading dataset {args.dataset} with arguments {args.dataset_kwargs}")
    dataset = dataset_class(**args.dataset_kwargs)
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    logger.debug(dataset[0])

    combined_output_file = os.path.join(args.output_dir, f"{args.dataset}_combined.jsonl")
    combined_grades = {}  # Dictionary to hold merged results: id -> dict

    start = time.time()
    for metric in args.metrics:
        if metric not in grader_configs["graders"]:
            logger.error(f"Metric {metric} not found in grader config. Available metrics: {list(grader_configs['graders'].keys())}. Skipping.")
            continue
        grader_params = grader_configs["graders"][metric]
        logger.debug(f"Grader params for {metric}: {grader_params}")
        m = "_".join(metric.split())
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{m}.jsonl")

        graded_responses = []
        
        # Overwrite is based on the existence of the metric intermediate file
        if os.path.exists(output_file) and not args.overwrite:
            logger.warning(f"Output file {output_file} for {metric} already exists and --overwrite not present. Loading existing results.")
            with jsonlines.open(output_file) as reader:
                graded_responses = list(reader)
        else:
            logger.info(f"Initializing grader for {metric}")
            grader = Grader.get_grader(metric, grader_params)
            graded_responses = await grader(dataset)
            logger.info(f"Done grading responses for {metric}")

            logger.info(f"Writing output to {output_file}")
            with jsonlines.open(output_file, "w") as writer:
                writer.write_all(graded_responses)
            logger.info(f"Finished writing output to {output_file}")

        # Update combined grades
        logger.info(f"Updating combined results with {metric}")
        for item in graded_responses:
            if item["id"] not in combined_grades:
                combined_grades[item["id"]] = {}
            combined_grades[item["id"]].update(item)

        # Prepare list for writing (inject metadata)
        # We rely on the fact that combined_grades keys (insertion order) align with dataset
        # because the first grader run (or load) populates it in dataset order.
        current_combined_list = []
        for datapoint, res in zip(dataset, combined_grades.values()):
            merged_item = res.copy()
            merged_item.update({
                "meta": datapoint.copy()
            })
            # Always inject completeness-human from dataset metadata when the grader was
            # not included in --metrics (e.g. Prometheus runs with explicit metric lists).
            if "completeness-human" not in merged_item:
                hs = datapoint.get("meta", {}).get("human_completeness_score", {})
                if hs and hs.get("score") is not None:
                    merged_item["completeness-human"] = {
                        "scoring_model": "Human",
                        "score": hs["score"],
                        "points": hs.get("points"),
                        "total_points": hs.get("total_points"),
                    }
            current_combined_list.append(merged_item)

        logger.info(f"Writing combined output to {combined_output_file}")
        with jsonlines.open(combined_output_file, "w") as writer:
            writer.write_all(current_combined_list)

    end = time.time()
    elapsed = timedelta(seconds=end - start)
    elapsed_str = str(elapsed).split(".")[0]  # e.g., '0:03:27' or '1 day, 2:03:27'
    logger.info(f"Grading completed in {elapsed_str}")


if __name__ == "__main__":
    # Use a single asyncio.run() to manage the entire async process
    asyncio.run(main())