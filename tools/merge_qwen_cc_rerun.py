"""Merge Qwen3.6 Criteria-Checklist MedExpert rerun results into the main rq2_analysis_results.jsonl.

Replaces all existing Qwen3.6 × Criteria-Checklist × MedExpert records in the main results file
with fresh records from the rerun output, then writes the merged file in-place (with a backup).

Usage:
    python tools/merge_qwen_cc_rerun.py \
        --main results/reframe/tables_media/qualitative_analysis/rq2_analysis_results.jsonl \
        --rerun results/reframe/tables_media/qualitative_analysis/qwen_cc_rerun/rq2_analysis_results.jsonl
"""
import argparse
import shutil
from pathlib import Path

import jsonlines


def is_qwen_cc_medexpert(rec: dict) -> bool:
    return (
        rec.get("Eval Model") == "Qwen3.6"
        and rec.get("Metric") == "Criteria-Checklist"
        and rec.get("Dataset") == "MedExpert"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Qwen CC MedExpert rerun into main results.")
    parser.add_argument("--main", required=True, help="Path to main rq2_analysis_results.jsonl")
    parser.add_argument("--rerun", required=True, help="Path to rerun rq2_analysis_results.jsonl")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a .bak backup")
    return parser.parse_args()


def main():
    args = parse_args()
    main_path = Path(args.main)
    rerun_path = Path(args.rerun)

    with jsonlines.open(main_path) as f:
        main_records = list(f)
    with jsonlines.open(rerun_path) as f:
        rerun_records = list(f)

    kept = [r for r in main_records if not is_qwen_cc_medexpert(r)]
    removed = len(main_records) - len(kept)

    rerun_errors = sum(1 for r in rerun_records if r.get("parse_error"))
    print(f"Main:  {len(main_records)} records total, {removed} Qwen CC MedExpert removed")
    print(f"Rerun: {len(rerun_records)} records ({rerun_errors} parse errors)")

    merged = kept + rerun_records
    print(f"Merged: {len(merged)} records total")

    if not args.no_backup:
        bak = main_path.with_suffix(".jsonl.bak")
        shutil.copy2(main_path, bak)
        print(f"Backup written to {bak}")

    with jsonlines.open(main_path, "w") as w:
        w.write_all(merged)
    print(f"Written to {main_path}")


if __name__ == "__main__":
    main()
