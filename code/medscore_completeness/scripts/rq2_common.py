"""Shared machinery for the RQ2 LLM classifiers.

Used by two entry points, each with its own prompt:
  - reasoning_alignment.py    TP rows    → prompt defined in the script         (Yes / Partially / No)
  - disagreement_analysis.py  FP/FN rows → prompts defined in the script, one per verdict type
                                           (disagreement labels derived in code)

Each entry point passes ``run_task`` a system prompt (or a dict of them keyed by verdict type) and
a function that builds the user prompt for one row.

Both read a results file (rq2_qualitative.jsonl, written by notebooks/analysis.py), keep the rows
whose verdict type matches, and send them to the classifier through one of two execution modes:
  1. Async streaming  — APIModel subclass with concurrent requests (default)
  2. OpenAI Batch API — --batch to submit, --resume to poll and retrieve. Rows are split into
     fixed-size chunks (default 5 000) so no batch file exceeds the API limit.

Output records keep the schema notebooks/analysis.py parses (``prompt_type`` is ``same_verdict``
for alignment rows and ``different_verdict`` for disagreement rows).
"""
import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import jsonlines
import openai
import tiktoken

from medscore_completeness.utils import redact_secrets
from medscore_completeness import configure_logging
from medscore_completeness.graders.LLMAPIModel import APIModel

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"

# Builds the user prompt for one row, given the row and its resolved verdict type.
UserPromptBuilder = Callable[[Dict[str, Any], str], str]

# JHU WSE Gateway (OpenAI-compatible). Setup follows LLM-Judge-Internal-Faithfulness
# (code/pipeline/demo_data/demo_gateway_config_gpt54_claude_cot_all.yaml):
#   - OpenAI models take the gateway's "openai/<model>" path. For this key only the undated alias is
#     allowed; it resolves to the same gpt-5.4-mini-2026-03-17 snapshot (probed 2026-11-11). The
#     served model is stored in each output record as ``served_model``.
#   - The gateway's WAF blocks the openai SDK's default User-Agent, so send a plain one.
#   - The key (GATEWAY_KEY in .env) has a ~60 requests/minute ceiling. The gateway has no Batch API.
GATEWAY_URL = "https://gateway.engineering.jhu.edu/gateway/compat"
GATEWAY_MODEL = "openai/gpt-5.4-mini"
GATEWAY_HEADERS = {"User-Agent": "python-requests/2.32.3"}

# ---------------------------------------------------------------------------
# Row preparation
# ---------------------------------------------------------------------------


def resolve_verdict_type(row: Dict[str, Any]) -> str:
    """Return TP/TN/FP/FN for a row, resolving cross-dataset 'Agrees'/'Disagrees' via the judgments.

    Cross-dataset rows carry analysis_category 'Agrees' or 'Disagrees', which the prompts do not
    define. Passing the raw category leaves the LLM to guess the direction (it sometimes swaps
    clinician and Judge). Unresolvable rows keep their raw category and are logged.
    """
    cat = row.get("analysis_category", "")
    if cat not in ("Agrees", "Disagrees"):
        return cat
    human_j = str(row.get("Human Judgment", "")).strip().lower()
    model_j = str(row.get("Model Judgment", "")).strip().lower()
    readable = all(j.startswith(("complete", "incomplete")) for j in (human_j, model_j))
    human_complete = human_j.startswith("complete")
    model_complete = model_j.startswith("complete")
    if readable and cat == "Agrees" and human_complete == model_complete:
        return "TN" if human_complete else "TP"
    if readable and cat == "Disagrees" and human_complete != model_complete:
        return "FP" if human_complete else "FN"
    logger.warning(
        f"Could not resolve {cat!r} for ID={row.get('ID')!r} "
        f"(Human Judgment={row.get('Human Judgment')!r}, Model Judgment={row.get('Model Judgment')!r})"
    )
    return cat


def make_row_id(row: Dict[str, Any], idx: int) -> str:
    return (
        f"{row.get('ID', '')}||{row.get('Eval Model', '')}||{row.get('Metric', '')}||"
        f"{row.get('analysis_category', '')}||{idx}"
    )


def load_rows(input_path: str, subset_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load the results file, drop HealthBench × Criteria-Checklist rows, optionally keep a subset."""
    with jsonlines.open(input_path) as reader:
        rows = list(reader)
    logger.info(f"Loaded {len(rows)} rows from {input_path}")

    n_before = len(rows)
    rows = [
        r for r in rows
        if not (
            r.get("Dataset") in {"HealthBench", "HealthBench-MT"}
            and r.get("Metric") in {"Criteria-Checklist", "Dynamic-Checklist"}
        )
    ]
    logger.info(f"Filtered {n_before - len(rows)} HealthBench/MT × Criteria-Checklist rows → {len(rows)} remaining")

    if subset_file:
        with jsonlines.open(subset_file) as reader:
            keep = {(r["ID"], r["Metric"], r["Eval Model"]) for r in reader}
        rows = [r for r in rows if (r.get("ID"), r.get("Metric"), r.get("Eval Model")) in keep]
        logger.info(f"Kept {len(rows)} rows matching {len(keep)} keys in {subset_file}")
    return rows


def prepare_rows(
    rows: List[Dict[str, Any]],
    verdict_types: Iterable[str],
    prompt_type: str,
    system_prompt: Union[str, Dict[str, str]],
    build_user_prompt: UserPromptBuilder,
    prompt_version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Keep rows whose resolved verdict type is in ``verdict_types`` and attach prompts.

    ``system_prompt`` is one prompt for every row, or a dict keyed by verdict type.

    The row index used in ``row_id`` is the position in the full input, so ids stay stable
    across runs with different filters.
    """
    verdict_types = set(verdict_types)
    prepared = []
    for idx, row in enumerate(rows):
        verdict_type = resolve_verdict_type(row)
        if verdict_type not in verdict_types:
            continue
        item = dict(row)
        item["verdict_type"] = verdict_type
        item["row_id"] = make_row_id(row, idx)
        item["prompt_type"] = prompt_type
        item["prompt_version"] = prompt_version
        item["_system_prompt"] = system_prompt[verdict_type] if isinstance(system_prompt, dict) else system_prompt
        item["_user_prompt"] = build_user_prompt(row, verdict_type)
        prepared.append(item)
    return prepared


def sample_per_verdict_type(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Sample up to ``n`` rows per verdict type (for prompt debugging)."""
    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    for vt in dict.fromkeys(r["verdict_type"] for r in rows):
        group = [r for r in rows if r["verdict_type"] == vt]
        sampled.extend(rng.sample(group, min(n, len(group))))
    return sampled


def build_output_record(
    item: Dict[str, Any],
    raw: str,
    parsed: Optional[Dict],
    parse_error: bool,
    scoring_model: str,
) -> Dict[str, Any]:
    """Build one output record from an item and its LLM response."""
    original_fields = {
        k: item[k]
        for k in (
            "analysis_category", "Dataset", "ID", "Metric", "Eval Model",
            "Question", "Answer", "Human Judgment", "Model Judgment",
            "Explanation of Human Judgment", "Explanation of Model Judgment",
            "Notes", "Omissions Aligned?",
        )
        if k in item
    }
    return {
        **original_fields,
        "verdict_type": item["verdict_type"],
        "row_id": item["row_id"],
        "prompt_type": item["prompt_type"],
        "prompt_version": item.get("prompt_version"),
        "full_system_prompt": item["_system_prompt"],
        "full_user_prompt": item["_user_prompt"],
        "model_response_raw": raw,
        "model_response_parsed": parsed,
        "parse_error": parse_error,
        "scoring_model": scoring_model,
    }


def _parse_json(raw: str) -> Tuple[Optional[Dict], bool]:
    # Claude models wrap JSON in ```json fences even in JSON mode; strip them.
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        return None, True


def _request_body(item: Dict[str, Any], model_name: str, max_completion_tokens: int,
                  seed: int, temperature: Optional[float]) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": item["_system_prompt"]},
            {"role": "user", "content": item["_user_prompt"]},
        ],
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
        "seed": seed,
    }
    if temperature is not None:
        body["temperature"] = temperature
    return body


# ---------------------------------------------------------------------------
# Async grader
# ---------------------------------------------------------------------------


class RQ2Grader(APIModel):
    """Async LLM grader for RQ2 rows."""

    def __init__(self, temperature: Optional[float] = None,
                 default_headers: Optional[Dict[str, str]] = None,
                 rpm: Optional[int] = None, **kwargs):
        self.temperature = temperature
        # Client-side pacing: every request, including retries, waits for its slot, so the
        # process never sends more than ``rpm`` requests per minute. Pacing is per process.
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()
        if default_headers is None:
            super().__init__(**kwargs)
            return
        # Gateway: APIModel would append /v1 to the URL and its connection test would send the
        # SDK's default User-Agent, which the gateway's WAF blocks. Build the client directly.
        super().__init__(skip_connection_test=True, **kwargs)
        self.client = openai.AsyncOpenAI(
            api_key=kwargs["api_key"],
            base_url=kwargs["model_server"],
            max_retries=0,
            default_headers=default_headers,
        )

    async def _wait_for_slot(self) -> None:
        if not self._min_interval:
            return
        async with self._pace_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def _individual_call(self, item: Dict[str, Any]):
        await self._wait_for_slot()
        body = _request_body(item, self.model_name, self.max_completion_tokens,
                             self.random_seed, self.temperature)
        return await self.client.chat.completions.create(**body)

    def _process_completions(self, batch, completions):
        results = []
        for item, completion in zip(batch, completions):
            raw, parsed, parse_error, served_model = "", None, False, None
            if completion is not None:
                raw = completion.choices[0].message.content or ""
                if raw.startswith("API_ERROR:"):
                    parse_error = True
                else:
                    parsed, parse_error = _parse_json(raw)
                    served_model = getattr(completion, "model", None)
            record = build_output_record(item, raw, parsed, parse_error, self.model_name)
            record["served_model"] = served_model
            results.append(record)
        return results


# ---------------------------------------------------------------------------
# Batch API runner
# ---------------------------------------------------------------------------


class RQ2BatchRunner:
    """Submits, polls, and retrieves OpenAI Batch API jobs.

    Rows are split by verdict type and chunked into fixed-size slices. Chunk labels are
    ``{verdict_type}_{chunk_idx}`` (e.g. ``FN_0``). State lives in ``batch_state_{task}.json``
    so the two tasks can share an output directory:

        {"batches": {"TP_0": {"batch_id": "...", "custom_id_map": {...}, "output_file_id": "..."}}}

    ``output_file_id`` is written back as soon as a batch completes, so ``--resume`` is idempotent.
    """

    def __init__(self, api_key: str, model_server: str, output_dir: str, task: str):
        self._output_dir = output_dir
        self.state_file = os.path.join(output_dir, f"batch_state_{task}.json")
        self._client = openai.OpenAI(api_key=api_key, base_url=model_server)

    def _save_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> Dict[str, Any]:
        with open(self.state_file) as f:
            return json.load(f)

    def _submit_one(self, requests: List[Dict], label: str) -> str:
        """Write, upload, and submit one batch file. Returns batch_id."""
        requests_path = os.path.join(self._output_dir, f"batch_requests_{label}.jsonl")
        with open(requests_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        logger.info(f"[{label}] Batch requests written to {requests_path} ({len(requests)} rows)")
        with open(requests_path, "rb") as f:
            uploaded = self._client.files.create(file=f, purpose="batch")
        batch = self._client.batches.create(
            input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
        )
        logger.info(f"[{label}] Batch submitted: id={batch.id}")
        return batch.id

    def submit_all(
        self,
        rows: List[Dict[str, Any]],
        model_name: str,
        max_completion_tokens: int,
        seed: int,
        temperature: Optional[float] = None,
        chunk_size: int = 5000,
    ) -> Dict[str, Any]:
        """Submit one batch per (verdict type, chunk), merging into any existing state."""
        state = self.load_state() if os.path.exists(self.state_file) else {"batches": {}}
        for vt in dict.fromkeys(r["verdict_type"] for r in rows):
            subset = [r for r in rows if r["verdict_type"] == vt]
            for chunk_idx in range(0, len(subset), chunk_size):
                chunk = subset[chunk_idx:chunk_idx + chunk_size]
                label = f"{vt}_{chunk_idx // chunk_size}"
                requests = [
                    {
                        "custom_id": item["row_id"],
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": _request_body(item, model_name, max_completion_tokens, seed, temperature),
                    }
                    for item in chunk
                ]
                state["batches"][label] = {
                    "batch_id": self._submit_one(requests, label),
                    "custom_id_map": {item["row_id"]: pos for pos, item in enumerate(chunk)},
                }
        self._save_state(state)
        logger.info(f"State saved to {self.state_file}")
        return state

    def poll_all(self, state: Dict[str, Any], poll_interval: int = 60) -> Dict[str, str]:
        """Poll all batches until done. Returns {label: output_file_id}."""
        output_file_ids: Dict[str, str] = {}
        pending: Dict[str, str] = {}
        for label, info in state["batches"].items():
            if "output_file_id" in info:
                logger.info(f"[{label}] Already completed — skipping poll")
                output_file_ids[label] = info["output_file_id"]
            else:
                pending[label] = info["batch_id"]

        while pending:
            for label in list(pending):
                batch = self._client.batches.retrieve(pending[label])
                counts = batch.request_counts
                logger.info(
                    f"[{label}] Batch {batch.id}: status={batch.status} "
                    f"({counts.completed}/{counts.total} completed, {counts.failed} failed)"
                )
                if batch.status in ("completed", "cancelled"):
                    if batch.status == "cancelled":
                        logger.warning(f"[{label}] Batch {batch.id} was cancelled — saving partial results")
                    if batch.output_file_id:
                        output_file_ids[label] = batch.output_file_id
                        state["batches"][label]["output_file_id"] = batch.output_file_id
                    else:
                        state["batches"][label]["cancelled"] = True
                    del pending[label]
                    self._save_state(state)
                elif batch.status in ("failed", "expired"):
                    raise RuntimeError(f"[{label}] Batch {batch.id} ended with status={batch.status!r}")
            if pending:
                time.sleep(poll_interval)
        return output_file_ids

    def retrieve_all(
        self,
        output_file_ids: Dict[str, str],
        state: Dict[str, Any],
        rows: List[Dict[str, Any]],
        model_name: str,
    ) -> List[Dict[str, Any]]:
        """Download and parse all batch outputs. Missing rows are emitted as parse errors."""
        rows_by_id = {r["row_id"]: r for r in rows}
        all_results: List[Dict[str, Any]] = []

        for label, info in state["batches"].items():
            if info.get("cancelled") and label not in output_file_ids:
                logger.warning(f"[{label}] Cancelled with no output file — {len(info['custom_id_map'])} rows missing")

        for label, output_file_id in output_file_ids.items():
            custom_id_map = state["batches"][label]["custom_id_map"]
            results_by_id: Dict[str, Dict] = {}
            for line in self._client.files.content(output_file_id).text.splitlines():
                if not line.strip():
                    continue
                result = json.loads(line)
                custom_id = result["custom_id"]
                item = rows_by_id.get(custom_id)
                if custom_id not in custom_id_map or item is None:
                    logger.warning(f"[{label}] No row for custom_id={custom_id!r} — skipping")
                    continue
                choices = ((result.get("response") or {}).get("body") or {}).get("choices", [])
                if choices:
                    raw = choices[0].get("message", {}).get("content", "") or ""
                    parsed, parse_error = _parse_json(raw)
                else:
                    raw, parsed, parse_error = f"Batch API error: {result.get('error') or {}}", None, True
                results_by_id[custom_id] = build_output_record(item, raw, parsed, parse_error, model_name)

            for row_id in custom_id_map:
                if row_id in results_by_id:
                    all_results.append(results_by_id[row_id])
                elif row_id in rows_by_id:
                    logger.warning(f"[{label}] No batch result for row_id={row_id!r}")
                    all_results.append(build_output_record(rows_by_id[row_id], "", None, True, model_name))
        return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser(description: str, default_output_file: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Results file (rq2_qualitative.jsonl from analysis.py)")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--output-file", default=default_output_file,
                        help=f"Output filename (default: {default_output_file})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--model-name", default=os.environ.get("MODEL"),
                        help=f"LLM model name (default: $MODEL, else {DEFAULT_MODEL}, or {GATEWAY_MODEL} with --gateway)")
    parser.add_argument("--model-server", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--gateway", action="store_true",
                        help=f"Use the JHU WSE Gateway ({GATEWAY_URL}) with $GATEWAY_KEY and model "
                             f"{GATEWAY_MODEL} unless --model-name is given. Async only (no --batch).")
    parser.add_argument("--api-key",
                        default=os.environ.get("GRADER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                        help="API key (default: $GRADER_API_KEY or $OPENAI_API_KEY)")

    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument("--batch", action="store_true", help="Submit OpenAI Batch API jobs")
    batch_group.add_argument("--resume", action="store_true",
                             help="Poll batches in the state file and retrieve results when complete")
    parser.add_argument("--batch-chunk-size", type=int, default=5000, help="Max rows per batch file")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between batch polls")
    parser.add_argument("--batch-size", type=int, default=10, help="Max concurrent requests (async path)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Max requests per minute for this process, retries included "
                             "(default: 50 with --gateway, else unlimited). Run one gateway job at a time.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (default: API default)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max completion tokens")
    parser.add_argument("--subset-file", default=None,
                        help="JSONL of rows to keep, matched on (ID, Metric, Eval Model), "
                             "e.g. ml4h_blind/rq2_blind_validation_key.jsonl")
    parser.add_argument("--n-examples", type=int, default=None,
                        help="Sample N rows per verdict type (prompt debugging)")
    parser.add_argument("--dry-run", action="store_true", help="Print row count and estimated tokens, then exit")
    parser.add_argument("--input-cost-per-1m", type=float, default=None, help="Input $/1M tokens (dry run)")
    parser.add_argument("--output-cost-per-1m", type=float, default=None, help="Output $/1M tokens (dry run)")
    parser.add_argument("--estimated-output-tokens", type=int, default=None,
                        help="Estimated output tokens per call for the dry run (default: --max-tokens)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def _print_dry_run(args: argparse.Namespace, rows: List[Dict[str, Any]], n_total: int) -> None:
    try:
        enc = tiktoken.encoding_for_model(args.model_name)
    except (KeyError, TypeError):
        enc = tiktoken.get_encoding("cl100k_base")
    input_tokens = sum(len(enc.encode(r["_system_prompt"])) + len(enc.encode(r["_user_prompt"])) for r in rows)
    per_call = args.estimated_output_tokens if args.estimated_output_tokens is not None else args.max_tokens
    output_tokens = len(rows) * per_call
    print(f"\n{'─' * 55}")
    print(f"  Dry-run estimate for: {args.model_name}")
    print(f"{'─' * 55}")
    print(f"  Rows after filters:    {n_total:>10,}")
    print(f"  Rows to process:       {len(rows):>10,}")
    print(f"  Total input tokens:    {input_tokens:>10,}")
    print(f"  Est. output tokens:    {output_tokens:>10,}  ({per_call} per call)")
    if args.input_cost_per_1m is not None and args.output_cost_per_1m is not None:
        cost_in = input_tokens / 1_000_000 * args.input_cost_per_1m
        cost_out = output_tokens / 1_000_000 * args.output_cost_per_1m
        print(f"  Estimated cost:        ${cost_in + cost_out:>9.4f}  (in ${cost_in:.4f}, out ${cost_out:.4f})")
    print(f"{'─' * 55}\n")


async def run_task(
    args: argparse.Namespace,
    task: str,
    verdict_types: Iterable[str],
    prompt_type: str,
    system_prompt: Union[str, Dict[str, str]],
    build_user_prompt: UserPromptBuilder,
    prompt_version: Optional[str] = None,
    postprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> None:
    """Load, filter, classify, and write results for one RQ2 task.

    ``postprocess`` is applied to each output record before writing (e.g. to add derived fields).
    """
    if args.gateway:
        if args.batch or args.resume:
            raise SystemExit("--gateway does not support the OpenAI Batch API; drop --batch/--resume.")
        args.model_server = GATEWAY_URL
        args.api_key = os.environ.get("GATEWAY_KEY")
        if not args.api_key:
            raise SystemExit("--gateway needs $GATEWAY_KEY (e.g. `set -a; . ./.env; set +a`).")
        args.model_name = args.model_name or GATEWAY_MODEL
        args.rpm = args.rpm or 50
    else:
        args.model_name = args.model_name or DEFAULT_MODEL
    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(
        log_level=logging.DEBUG if args.debug else logging.INFO,
        log_file=os.path.join(args.output_dir, f"{task}.log"),
        filemode="w" if args.overwrite else "a",
    )
    logger.debug(f"Parsed arguments: {redact_secrets(vars(args))}")

    output_file = os.path.join(args.output_dir, args.output_file)
    if os.path.exists(output_file) and not args.overwrite and not args.dry_run:
        logger.warning(f"Output file {output_file} already exists and --overwrite not set. Skipping.")
        return

    logger.info(f"Prompt version: {prompt_version}")

    all_rows = load_rows(args.input, args.subset_file)
    rows = prepare_rows(all_rows, verdict_types, prompt_type, system_prompt, build_user_prompt, prompt_version)
    if args.n_examples is not None:
        rows = sample_per_verdict_type(rows, args.n_examples, args.seed)
    counts = {vt: sum(r["verdict_type"] == vt for r in rows) for vt in verdict_types}
    logger.info(f"Prepared {len(rows)} rows for {task}: {counts}")

    if args.dry_run:
        _print_dry_run(args, rows, len(all_rows))
        return

    with open(os.path.join(args.output_dir, f"args_{task}.json"), "w") as f:
        json.dump(redact_secrets(vars(args)), f, indent=4)

    if args.batch or args.resume:
        runner = RQ2BatchRunner(args.api_key, args.model_server, args.output_dir, task)
        if args.resume:
            logger.info(f"Resuming batches from {runner.state_file}")
            state = runner.load_state()
        else:
            state = runner.submit_all(rows, args.model_name, args.max_tokens, args.seed,
                                      args.temperature, args.batch_chunk_size)
        logger.info(f"Polling batches every {args.poll_interval}s...")
        output_file_ids = runner.poll_all(state, poll_interval=args.poll_interval)
        results = runner.retrieve_all(output_file_ids, state, rows, args.model_name)
    else:
        grader = RQ2Grader(
            temperature=args.temperature,
            default_headers=GATEWAY_HEADERS if args.gateway else None,
            rpm=args.rpm,
            model_name=args.model_name,
            model_server=args.model_server,
            api_key=args.api_key,
            max_concurrent_requests=args.batch_size,
            max_completion_tokens=args.max_tokens,
            random_seed=args.seed,
        )
        start = time.time()
        results = await grader(rows)
        logger.info(f"Grading completed in {time.time() - start:.1f}s")

    if postprocess is not None:
        results = [postprocess(r) for r in results]
    n_errors = sum(1 for r in results if r.get("parse_error"))
    logger.info(f"Writing {len(results)} results to {output_file} ({n_errors} parse errors)")
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(results)
