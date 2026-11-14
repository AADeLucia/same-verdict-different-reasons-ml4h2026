"""Extract the MedExpert double-annotated (IAA) responses as clinician pairs.

The published dataset (`sonal-ssj/MedExpert`, config `medexpert-all`) has 572 rows: the 540
benchmark rows plus 32 second annotations. All 64 rows of the 32 doubly annotated responses carry
`iaa=True` and share a `base_title`; exactly one row of each pair is in the 540-row benchmark (the
clinician label used in the paper, "clinician A"), and the other is the second annotation
("clinician B").

Output: one JSONL row per pair, keyed on the benchmark `id` so it joins to the graded results.

Usage:
    PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" python tools/extract_medexpert_double_annotations.py \
        --output-file results/reframe/tables_media/ml4h_rebuttal/medexpert_double_annotations.jsonl
"""
import argparse
import logging
import os

import jsonlines
from datasets import load_dataset

from medscore_completeness.utils.medical_qa_dataset import MEDEXPERT_OMISSION_SEVERITY_TO_SCORE

# Paper's primary ("relaxed") clinician label: Complete if the severity score is >= 0.75
# (no omissions or only Mild ones). Mirrors create_model_human_comparison_df in notebooks/analysis.py.
RELAXED_COMPLETE_THRESHOLD = 0.75


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", default="sonal-ssj/MedExpert")
    parser.add_argument("--all-config", default="medexpert-all")
    parser.add_argument("--benchmark-config", default="medexpert-benchmark")
    parser.add_argument(
        "--output-file",
        default=os.path.join(
            os.environ.get("PROJECT_DIR", "."),
            "results/reframe/tables_media/ml4h_rebuttal/medexpert_double_annotations.jsonl",
        ),
    )
    return parser.parse_args()


def clinician_fields(row, prefix):
    severity = row["omission_severity"]
    score = MEDEXPERT_OMISSION_SEVERITY_TO_SCORE[severity]
    return {
        f"{prefix}_id": row["id"],
        f"{prefix}_title": row["title"],
        f"{prefix}_annotator": row["annotator"],
        f"{prefix}_severity": severity.split(" - ")[0] if severity else "None",
        f"{prefix}_score": score,
        f"{prefix}_judgment": "Complete" if score >= RELAXED_COMPLETE_THRESHOLD else "Incomplete",
        f"{prefix}_n_omissions": int(row["n_omissions"]),
        f"{prefix}_omissions": [str(o) for o in row["omissions"]] if row["omissions"] is not None else [],
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    all_df = load_dataset(args.dataset_name, args.all_config)["train"].to_pandas()
    benchmark_ids = set(load_dataset(args.dataset_name, args.benchmark_config)["train"]["id"])
    iaa_df = all_df[all_df["iaa"]]

    pairs = []
    for base_title, group in iaa_df.groupby("base_title"):
        in_benchmark = group[group["id"].isin(benchmark_ids)]
        second = group[~group["id"].isin(benchmark_ids)]
        if len(group) != 2 or len(in_benchmark) != 1 or len(second) != 1:
            logger.warning(f"Skipping {base_title}: unexpected pair structure ({len(group)} rows)")
            continue
        a, b = in_benchmark.iloc[0], second.iloc[0]
        pairs.append({
            "id": a["id"],
            "base_title": base_title,
            "domain": a["domain"],
            "model": a["model"],
            "same_question": a["question"] == b["question"],
            "same_response": a["response"] == b["response"],
            **clinician_fields(a, "clinician_a"),
            **clinician_fields(b, "clinician_b"),
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with jsonlines.open(args.output_file, "w") as writer:
        writer.write_all(pairs)
    n_agree = sum(p["clinician_a_judgment"] == p["clinician_b_judgment"] for p in pairs)
    logger.info(f"Wrote {len(pairs)} pairs to {args.output_file} (binary agreement {n_agree}/{len(pairs)})")


if __name__ == "__main__":
    main()
