"""Pre-compute HealthBench human completeness scores.

Grades each response against the human-defined rubric criteria embedded in the
dataset, then writes an augmented JSONL with meta["human_completeness_score"]
populated. This score is static — it does not change with the backbone model
under evaluation and should be computed once with a fixed strong model.

Usage (async/streaming):
    python -m medscore_completeness.scripts.compute_healthbench_human_scores \\
        --model-name meta-llama/Llama-3.3-70B-Instruct \\
        --model-server https://api.together.xyz/v1 \\
        --output-dir ./results/healthbench_human_scores

Usage (OpenAI Batch API — cheaper for GPT models):
    # Submit a new batch:
    python -m medscore_completeness.scripts.compute_healthbench_human_scores \\
        --batch \\
        --model-name gpt-4.1 \\
        --api-key $OPENAI_API_KEY \\
        --output-dir ./results/healthbench_human_scores

    # Resume an existing batch (skips re-upload):
    python -m medscore_completeness.scripts.compute_healthbench_human_scores \\
        --batch-id batch_abc123 \\
        --model-name gpt-4.1 \\
        --api-key $OPENAI_API_KEY \\
        --output-dir ./results/healthbench_human_scores

The output JSONL can then be passed as --dataset-path to HealthBenchDataset so
that get_human_completeness_score() returns the pre-computed value from meta.
"""
import os
import argparse
import json
import logging
import asyncio
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import openai
import tiktoken
import jsonlines

from medscore_completeness.utils import redact_secrets
from medscore_completeness import configure_logging
from medscore_completeness.utils.medical_qa_dataset import HealthBenchDataset
from medscore_completeness.graders.healthbench_response_graders import (
    HealthBenchResponseGrader,
    RubricItem,
)


############
# Grader
############

GRADER_KEY = "completeness-healthbench-human"


class _PerItemCriteriaGrader(HealthBenchResponseGrader):
    """HealthBenchResponseGrader variant that uses per-item human-defined criteria.

    Criteria are supplied as a dict keyed by item ID and reconstructed from the
    criterion strings and point values stored in the dataset item.
    Not registered with the Grader registry (no grader_name class attribute).
    """

    def __init__(self, criteria_by_id: Dict[str, List[Dict]], **kwargs):
        self._criteria_by_id = {str(k): v for k, v in criteria_by_id.items()}
        # Set grader_name on the instance so HealthBenchResponseGrader.__call__
        # can use it as the output dict key without triggering auto-registration.
        self.grader_name = GRADER_KEY
        super().__init__(**kwargs)

    def get_criteria(self, item_id: Optional[str]) -> List[Dict[str, Any]]:
        return self._criteria_by_id.get(str(item_id), [])


def _build_rubric_items(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct rubric dicts from dataset item criteria strings + point values.

    HealthBenchDataset._load_data() stores:
      - item["criteria"]: List[str]  — criterion text only
      - item["meta"]["criteria_meta"]["points"]: List[float]  — one per criterion
    """
    criteria_strings = item.get("criteria", [])
    criteria_meta = item.get("meta", {}).get("criteria_meta", {})
    points_list = criteria_meta.get("points", [])

    rubric_items = []
    for criterion, points in zip(criteria_strings, points_list):
        rubric_items.append({
            "criterion": criterion,
            "points": points,
            "tags": [],
        })

    if not rubric_items:
        logging.getLogger(__name__).warning(
            f"Item {item.get('id')} has no criteria — will receive score=None"
        )
    return rubric_items


############
# Batch API
############

class BatchRunner:
    """Submits, polls, and retrieves OpenAI Batch API jobs.

    Uses the synchronous OpenAI client — batch operations don't require
    concurrency within a single run.
    """

    SEP = "|||"

    def __init__(
        self,
        grader: _PerItemCriteriaGrader,
        api_key: str,
        model_server: str,
        output_dir: str,
    ):
        self._grader = grader
        self._output_dir = output_dir
        self._state_file = os.path.join(output_dir, "batch_state.json")
        self._client = openai.OpenAI(api_key=api_key, base_url=model_server)

    def build_requests(
        self, dataset: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Build Batch API request objects and a custom_id → metadata map.

        Returns (requests, custom_id_map) so the map can be persisted to disk.
        """
        augmented = self._grader._augment_inputs(dataset)
        requests = []
        custom_id_map: Dict[str, Dict] = {}

        for idx, item in enumerate(augmented):
            item_id = str(item["id"])
            custom_id = f"{item_id}{self.SEP}{idx}"

            formatted_input = self._grader._format_input(item)
            sys_prompt = self._grader._get_system_prompt()
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": formatted_input},
            ]
            body: Dict[str, Any] = {
                "model": self._grader.model_name,
                "messages": messages,
                "max_completion_tokens": self._grader.max_completion_tokens,
                "response_format": {"type": "json_object"},
                "seed": self._grader.random_seed,
            }
            if self._grader.temperature is not None:
                body["temperature"] = self._grader.temperature
            if self._grader.top_p is not None:
                body["top_p"] = self._grader.top_p

            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            })
            custom_id_map[custom_id] = {
                "item_id": item_id,
                "rubric_item": item["rubric_item"].to_dict(),
            }

        return requests, custom_id_map

    def submit(self, requests: List[Dict], custom_id_map: Dict[str, Dict]) -> str:
        """Upload requests JSONL and create a batch. Persists state; returns batch_id."""
        log = logging.getLogger(__name__)

        requests_path = os.path.join(self._output_dir, "batch_requests.jsonl")
        with open(requests_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        log.info(f"Batch requests saved to {requests_path}")

        with open(requests_path, "rb") as f:
            uploaded = self._client.files.create(file=f, purpose="batch")

        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        state = {"batch_id": batch.id, "custom_id_map": custom_id_map}
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

        log.info(
            f"Batch submitted: id={batch.id}, requests={len(requests)}, "
            f"state → {self._state_file}"
        )
        return batch.id

    def poll(self, batch_id: str, poll_interval: int = 60) -> str:
        """Block until the batch completes. Returns output_file_id."""
        log = logging.getLogger(__name__)
        while True:
            batch = self._client.batches.retrieve(batch_id)
            counts = batch.request_counts
            log.info(
                f"Batch {batch_id}: status={batch.status} "
                f"({counts.completed}/{counts.total} completed, {counts.failed} failed)"
            )
            if batch.status == "completed":
                return batch.output_file_id
            if batch.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(
                    f"Batch {batch_id} ended with status={batch.status!r}"
                )
            time.sleep(poll_interval)

    def retrieve(
        self, output_file_id: str, custom_id_map: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Download output file and parse into {item_id: grader_result} format."""
        log = logging.getLogger(__name__)
        content = self._client.files.content(output_file_id)
        grader_result_by_id: Dict[str, Dict] = {}

        for line in content.text.splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            meta = custom_id_map.get(custom_id)
            if meta is None:
                log.warning(f"Unknown custom_id {custom_id!r} — skipping")
                continue

            item_id = meta["item_id"]
            rubric_item = RubricItem.from_dict(meta["rubric_item"])

            response_body = (result.get("response") or {}).get("body") or {}
            choices = response_body.get("choices", [])
            if choices:
                raw_content = choices[0].get("message", {}).get("content", "")
                try:
                    rubric_grade = json.loads(raw_content)
                except json.JSONDecodeError:
                    log.warning(f"JSON decode error for {custom_id}: {raw_content!r}")
                    rubric_grade = {"explanation": f"Error decoding: {raw_content!r}"}
            else:
                error = result.get("error") or {}
                rubric_grade = {"explanation": f"Batch API error: {error}"}

            if item_id not in grader_result_by_id:
                grader_result_by_id[item_id] = {
                    GRADER_KEY: {
                        "total_points": 0,
                        "score": 0,
                        "scoring_model": self._grader.model_name,
                        "meta": [],
                    }
                }
            entry = grader_result_by_id[item_id][GRADER_KEY]
            entry["total_points"] += rubric_item.points
            details = rubric_item.to_dict()
            details.update(rubric_grade)
            entry["meta"].append(details)
            entry["score"] += self._grader.parse_criteria_score(
                rubric_grade.get("criteria_score", False)
            )

        return grader_result_by_id


############
# Score merging (shared between async and batch paths)
############
def _merge_scores(
    results_by_id: Dict[str, Any],
    dataset,
    model_name: str,
    logger: logging.Logger,
) -> Tuple[List[Dict], int]:
    """Merge grader output into dataset items, computing human completeness scores.

    Returns (augmented_items, n_scored).
    """
    augmented = []
    n_scored = 0
    for item in dataset:
        item_id = str(item["id"])
        grader_result = results_by_id.get(item_id, {}).get(GRADER_KEY, {})

        criteria_results = grader_result.get("meta", [])
        logger.info(grader_result)

        # Weighted score: sum of earned points / sum of rubric point values
        total_points = grader_result.get("total_points")
        raw_score = sum(
            c.get("points", 0) for c in criteria_results if c.get("criteria_score") is True
        )
        if raw_score is not None and total_points:
            human_completeness_score = raw_score / total_points
            n_scored += 1
        else:
            human_completeness_score = None
            logger.warning(f"No grader output for item {item_id}")

        # Unweighted score: fraction of criteria met, treating each criterion as 1 point
        n_criteria = len(criteria_results)
        if n_criteria > 0:
            n_met = sum(1 for c in criteria_results if c.get("criteria_score") is True)
            human_completeness_score_unweighted = n_met / n_criteria
        else:
            human_completeness_score_unweighted = None

        augmented_item = item.copy()
        augmented_item["meta"] = item.get("meta", {}).copy()
        augmented_item["meta"]["human_completeness_score"] = human_completeness_score
        augmented_item["meta"]["human_completeness_score_unweighted"] = human_completeness_score_unweighted
        augmented_item["meta"]["human_completeness_score_unweighted_total"] = len(criteria_results)
        augmented_item["meta"]["human_completeness_score_weighted_total"] = total_points
        augmented_item["meta"]["human_completeness_score_details"] = criteria_results
        augmented_item["meta"]["human_completeness_scoring_model"] = model_name
        augmented.append(augmented_item)
    return augmented, n_scored


############
# Main
############
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-compute HealthBench human completeness scores by grading responses "
            "against the human-defined rubric criteria in the dataset, then write an "
            "augmented JSONL with meta['human_completeness_score'] populated."
        )
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output JSONL and log file"
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Output JSONL filename (default: healthbench_human_scores.jsonl)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file instead of skipping"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Estimate token usage and cost without making any API calls. "
            "Prints per-item and total token counts then exits."
        )
    )
    parser.add_argument(
        "--input-cost-per-1m", type=float, default=None,
        help="Input token cost per 1M tokens in USD (optional, for cost estimate)"
    )
    parser.add_argument(
        "--output-cost-per-1m", type=float, default=None,
        help="Output token cost per 1M tokens in USD (optional, for cost estimate)"
    )
    parser.add_argument(
        "--estimated-output-tokens", type=int, default=150,
        help="Estimated output tokens per rubric-item call for dry-run (default: 150)"
    )

    # Batch API options
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch", action="store_true",
        help=(
            "Use the OpenAI Batch API (50%% cheaper, up to 24h turnaround). "
            "Submits a new batch and polls until complete."
        )
    )
    batch_group.add_argument(
        "--batch-id", type=str, default=None,
        help=(
            "Resume an existing batch by ID (reads custom_id_map from "
            "batch_state.json in --output-dir). Skips upload/submit."
        )
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between batch status polls (default: 60)"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset-kwargs", type=json.loads, default="{}",
        help=(
            "JSON string of optional arguments for HealthBenchDataset "
            "(e.g. '{\"include_multiturn\": true}')"
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Grader settings (mirrored from healthbench-icl_verification.py)
    parser.add_argument(
        "--model-name", type=str, default=os.environ.get("MODEL"),
        help="LLM model name"
    )
    parser.add_argument(
        "--model-server", type=str, default=os.environ.get("SERVER"),
        help="LLM server URL (must end in /v1)"
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("GRADER_API_KEY"),
        help="LLM API key"
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 10)),
        help="Max concurrent API requests (async mode only)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Omit to use the API model's default (required for some models that do not accept a temperature parameter)."
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Top-p nucleus sampling. Omit to use the API model's default."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", 2048)),
        help="Max completion tokens per rubric item"
    )

    return parser.parse_args()


############
# Main
############
async def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configure_logging(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        log_file=os.path.join(args.output_dir, "grading.log"),
        filemode="w" if args.overwrite else "a",
    )
    logger = logging.getLogger(__name__)
    logger.debug(f"Parsed arguments: {args}")

    output_filename = args.output_file or "healthbench_human_scores.jsonl"
    output_file = os.path.join(args.output_dir, output_filename)

    if os.path.exists(output_file) and not args.overwrite:
        logger.warning(f"Output file {output_file} already exists and --overwrite not set. Skipping.")
        return

    # Persist args for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(redact_secrets(vars(args)), f, indent=4)

    # Load dataset
    dataset_kwargs = args.dataset_kwargs.copy()
    if "random_state" not in dataset_kwargs:
        dataset_kwargs["random_state"] = args.seed

    logger.info(f"Loading HealthBenchDataset with arguments {dataset_kwargs}")
    dataset = HealthBenchDataset(**dataset_kwargs)
    logger.info(f"Loaded {len(dataset)} examples")

    # Build per-item criteria lookup
    criteria_by_id: Dict[str, List[Dict]] = {}
    n_empty = 0
    for item in dataset:
        rubric_items = _build_rubric_items(item)
        criteria_by_id[str(item["id"])] = rubric_items
        if not rubric_items:
            n_empty += 1

    logger.info(f"Built criteria lookup for {len(criteria_by_id)} items ({n_empty} with no criteria)")

    # Dry-run: count tokens and estimate cost without making API calls
    if args.dry_run:
        try:
            enc = tiktoken.encoding_for_model(args.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        # Instantiate grader without hitting the network
        dry_grader = _PerItemCriteriaGrader(
            criteria_by_id=criteria_by_id,
            model_name=args.model_name,
            model_server=args.model_server or "https://api.openai.com/v1",
            api_key=args.api_key or "dry-run",
            skip_connection_test=True,
        )
        augmented = dry_grader._augment_inputs(list(dataset))
        n_calls = len(augmented)

        sys_prompt = dry_grader._get_system_prompt()
        sys_tokens = len(enc.encode(sys_prompt))

        total_input_tokens = 0
        for item in augmented:
            prompt = dry_grader._format_input(item)
            total_input_tokens += sys_tokens + len(enc.encode(prompt))

        total_output_tokens = n_calls * args.estimated_output_tokens

        print(f"\n{'─' * 50}")
        print(f"  Dry-run estimate for: {args.model_name}")
        print(f"{'─' * 50}")
        print(f"  Dataset items:         {len(dataset):>10,}")
        print(f"  Rubric-item calls:     {n_calls:>10,}")
        print(f"  Total input tokens:    {total_input_tokens:>10,}")
        print(f"  Est. output tokens:    {total_output_tokens:>10,}  ({args.estimated_output_tokens} per call)")
        print(f"  Total tokens:          {total_input_tokens + total_output_tokens:>10,}")
        if args.input_cost_per_1m is not None and args.output_cost_per_1m is not None:
            input_cost = total_input_tokens / 1_000_000 * args.input_cost_per_1m
            output_cost = total_output_tokens / 1_000_000 * args.output_cost_per_1m
            print(f"{'─' * 50}")
            print(f"  Input cost:            ${input_cost:>9.4f}")
            print(f"  Output cost:           ${output_cost:>9.4f}")
            print(f"  Total estimated cost:  ${input_cost + output_cost:>9.4f}")
        else:
            print(f"\n  (Pass --input-cost-per-1m and --output-cost-per-1m for cost estimate)")
        print(f"{'─' * 50}\n")
        return

    # -------------------------------------------------------------------------
    # OpenAI Batch API path
    # -------------------------------------------------------------------------
    if args.batch or args.batch_id:
        model_server = args.model_server or "https://api.openai.com/v1"

        grader = _PerItemCriteriaGrader(
            criteria_by_id=criteria_by_id,
            model_name=args.model_name,
            model_server=model_server,
            api_key=args.api_key,
            skip_connection_test=True,
        )
        runner = BatchRunner(
            grader=grader,
            api_key=args.api_key,
            model_server=model_server,
            output_dir=args.output_dir,
        )

        if args.batch_id:
            batch_id = args.batch_id
            logger.info(f"Resuming batch {batch_id} — loading state from {runner._state_file}")
            with open(runner._state_file) as f:
                state = json.load(f)
            custom_id_map = state["custom_id_map"]
        else:
            requests, custom_id_map = runner.build_requests(list(dataset))
            logger.info(f"Built {len(requests)} batch requests")
            batch_id = runner.submit(requests, custom_id_map)

        logger.info(f"Polling batch {batch_id} every {args.poll_interval}s...")
        output_file_id = runner.poll(batch_id, poll_interval=args.poll_interval)

        logger.info(f"Retrieving results from file {output_file_id}")
        results_by_id = runner.retrieve(output_file_id, custom_id_map)

    # -------------------------------------------------------------------------
    # Async streaming path
    # -------------------------------------------------------------------------
    else:
        grader_kwargs = dict(
            criteria_by_id=criteria_by_id,
            model_name=args.model_name,
            model_server=args.model_server,
            api_key=args.api_key,
            max_concurrent_requests=args.batch_size,
            max_completion_tokens=args.max_tokens,
        )
        if args.temperature is not None:
            grader_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            grader_kwargs["top_p"] = args.top_p

        grader = _PerItemCriteriaGrader(**grader_kwargs)
        logger.info(f"Grader: model={args.model_name}, server={args.model_server}")

        start = time.time()
        logger.info("Starting grading...")
        graded_responses = await grader(dataset)
        elapsed = str(timedelta(seconds=time.time() - start)).split(".")[0]
        logger.info(f"Grading completed in {elapsed}")

        results_by_id = {str(r["id"]): r for r in graded_responses}

    # -------------------------------------------------------------------------
    # Shared: merge scores and write output
    # -------------------------------------------------------------------------
    augmented, n_scored = _merge_scores(results_by_id, dataset, args.model_name, logger)
    logger.info(f"Scored {n_scored}/{len(dataset)} items")

    logger.info(f"Writing augmented dataset to {output_file}")
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(augmented)
    logger.info(f"Done. {n_scored} human completeness scores written to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
