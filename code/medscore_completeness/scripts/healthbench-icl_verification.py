"""Verify HealthBench-ICL Grader performance against known answers."""
import os
import argparse
import json
import logging
import asyncio
import time
from datetime import timedelta

import jsonlines

from medscore_completeness.utils import redact_secrets
from medscore_completeness import configure_logging
from medscore_completeness.utils.medical_qa_dataset import HealthBenchDataset
from medscore_completeness.graders import Grader


############
# Main
############
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade responses using HealthBench ICL Grader.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files instead of skipping them.")

    # Dataset settings
    parser.add_argument("--dataset-kwargs", type=json.loads, default="{}", help="JSON string of optional arguments for the dataset class (e.g. '{\"include_multiturn\": true}')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Grader settings
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL"), help="LLM Model Name")
    parser.add_argument("--model-server", type=str, default=os.environ.get("SERVER"), help="LLM Server URL")
    parser.add_argument("--api-key", type=str, default=os.environ.get("TOGETHER_API_KEY"), help="LLM API Key")
    parser.add_argument("--num-examples", type=int, default=2, help="Number of ICL examples to use")
    parser.add_argument("--similarity-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model for similarity")
    parser.add_argument("--cache-embeddings", action="store_true", help="Whether to cache embeddings for ICL")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("CACHE_DIR"), help="Directory to cache embeddings")

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

    # Save args
    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(redact_secrets(vars(args)), f, indent=4)

    # Load dataset for inference
    logger.info(f"Loading HealthBench dataset with arguments {args.dataset_kwargs}")
    if "seed" not in args.dataset_kwargs:
        args.dataset_kwargs["random_state"] = args.seed

    dataset = HealthBenchDataset(**args.dataset_kwargs)
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Move "criteria" to "metadata"
    for item in dataset:
        if "criteria" in item:
            item["criteria_ground_truth"] = item.pop("criteria")
            item["criteria"] = []
    logger.debug(f"Dataset example: {dataset[0]}")

    grader_name = "completeness-icl-healthbench"
    grader_params = dict(
        num_examples=args.num_examples,
        similarity_model=args.similarity_model,
        cache_embeddings=args.cache_embeddings,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        model_server=args.model_server,
        api_key=args.api_key
    )
    grader = Grader.get_grader(grader_name, grader_params)
    output_file = os.path.join(args.output_dir, f"{grader_name}.jsonl")

    if os.path.exists(output_file) and not args.overwrite:
        logger.warning(f"Output file {output_file} already exists and --overwrite not present. Skipping grading.")
        return

    start = time.time()
    logger.info(f"Initializing grader: {grader_name}")
    graded_responses = await grader(dataset)
    logger.info(f"Done grading responses for {grader_name}")

    # Add dataset metadata to graded responses
    combined_output = []
    for datapoint, res in zip(dataset, graded_responses):
        merged_item = res.copy()
        merged_item.update({
            "meta": datapoint.copy()
        })
        combined_output.append(merged_item)

    logger.info(f"Writing output to {output_file}")
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(combined_output)
    logger.info(f"Finished writing output to {output_file}")

    end = time.time()
    elapsed = timedelta(seconds=end - start)
    elapsed_str = str(elapsed).split(".")[0]
    logger.info(f"Grading completed in {elapsed_str}")


if __name__ == "__main__":
    asyncio.run(main())