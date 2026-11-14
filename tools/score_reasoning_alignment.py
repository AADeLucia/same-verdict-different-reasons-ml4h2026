"""Score reasoning_alignment.py runs against the blinded human TP labels.

Joins each run's Yes/Partially/No labels to the 30 human-labeled TP items on (ID, Metric, Eval Model)
and reports exact agreement, Cohen's kappa, Krippendorff's alpha (nominal, simpledorff; same as
notebooks/analysis.py), the confusion matrix, a GPT-5 Mini vs. other-Judge split, and the items
whose label differs from the human label. ``stored`` scores the labels in the blind key (the
classifier labels currently in the paper).

Usage:
    python tools/score_reasoning_alignment.py stored \\
        results/reframe/tables_media/qualitative_analysis/reasoning_alignment_validation/v0_original.jsonl
"""
import argparse
import os

import numpy as np
import pandas as pd
import simpledorff
from sklearn.metrics import cohen_kappa_score

BLIND_DIR = "results/reframe/tables_media/qualitative_analysis/ml4h_blind"
LABELS = ["Yes", "Partially", "No"]
KEY_COLS = ["ID", "Metric", "Eval Model"]


def load_validation(annotated_path: str, key_path: str) -> pd.DataFrame:
    ann = pd.read_json(annotated_path, lines=True)
    key = pd.read_json(key_path, lines=True)
    df = key.merge(ann[["item_id", "human_label", "human_notes"]], on="item_id", validate="1:1")
    df = df[(df["item_type"] == "tp_alignment") & ~df["is_retest"]]
    return df[df["human_label"].isin(LABELS)].reset_index(drop=True)


def load_run(path: str, derived: bool = False) -> pd.DataFrame:
    run = pd.read_json(path, lines=True)
    if derived:
        # v4+: label recomputed in code from per-concern matches (reasoning_alignment.derive_label)
        if "omissions_aligned_derived" not in run:
            raise SystemExit(f"{path} has no omissions_aligned_derived column (pre-v4 run?)")
        run["run_label"] = run["omissions_aligned_derived"]
    else:
        run["run_label"] = run["model_response_parsed"].map(
            lambda p: p.get("omissions_aligned") if isinstance(p, dict) else None
        )
    run["run_explanation"] = run["model_response_parsed"].map(
        lambda p: p.get("explanation") if isinstance(p, dict) else None
    )
    return run[KEY_COLS + ["run_label", "run_explanation"]]


def agreement(a: pd.Series, b: pd.Series) -> dict:
    mask = a.isin(LABELS) & b.isin(LABELS)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": len(a), "agree": np.nan, "kappa": np.nan, "alpha": np.nan}
    melted = pd.concat([
        pd.DataFrame({"item": a.index, "rater": "a", "label": a.values}),
        pd.DataFrame({"item": b.index, "rater": "b", "label": b.values}),
    ])
    try:
        alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
            melted, experiment_col="item", annotator_col="rater", class_col="label")
    except ZeroDivisionError:
        alpha = np.nan
    return {"n": len(a), "agree": (a == b).mean(), "kappa": cohen_kappa_score(a, b), "alpha": alpha}


def score(df: pd.DataFrame, name: str, show_errors: bool) -> dict:
    h, r = df["human_label"], df["run_label"]
    overall = agreement(h, r)
    gpt = df["Eval Model"] == "GPT-5 Mini"
    print(f"\n=== {name} ===")
    print(f"missing/unparsed labels: {(~r.isin(LABELS)).sum()}")
    print(f"overall   n={overall['n']}  agree={overall['agree']:.3f}  kappa={overall['kappa']:.3f}  alpha={overall['alpha']:.3f}")
    splits = [("GPT-5 Mini", gpt), ("other", ~gpt)]
    if "annotation_stage" in df:  # combined files (e.g. *_tp40*) mark stage 1 / stage 2 labels
        splits += [(stage, df["annotation_stage"] == stage) for stage in sorted(df["annotation_stage"].dropna().unique())]
    for label, mask in splits:
        s = agreement(h[mask], r[mask])
        print(f"{label:<10}n={s['n']}  agree={s['agree']:.3f}  kappa={s['kappa']:.3f}  alpha={s['alpha']:.3f}")
    print("confusion (rows = human, cols = run):")
    print(pd.crosstab(h, r).reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())
    if show_errors:
        rank = {lab: i for i, lab in enumerate(LABELS)}
        for _, row in df[h != r].iterrows():
            direction = "over" if rank.get(row["run_label"], 9) < rank[row["human_label"]] else "under"
            print(f"  {row['item_id']} human={row['human_label']:<9} run={row['run_label']!s:<9} ({direction}-credit)"
                  f"  {row['Eval Model']} / {row['Metric']}")
    return {"run": name, **overall}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="reasoning_alignment.py output files, or 'stored'")
    parser.add_argument("--annotated", default=os.path.join(BLIND_DIR, "rq2_blind_validation_annotator1_reduced_annotated.jsonl"))
    parser.add_argument("--key", default=os.path.join(BLIND_DIR, "rq2_blind_validation_key_reduced.jsonl"))
    parser.add_argument("--errors", action="store_true", help="List items that disagree with the human label")
    parser.add_argument("--derived", action="store_true",
                        help="Score omissions_aligned_derived (label recomputed in code, v4+) instead of the model's label")
    args = parser.parse_args()

    val = load_validation(args.annotated, args.key)
    summary = []
    for run_path in args.runs:
        if run_path == "stored":
            df = val.assign(run_label=val["classifier_label"])
        else:
            df = val.merge(load_run(run_path, args.derived), on=KEY_COLS, how="left", validate="1:1")
        summary.append(score(df, run_path if run_path == "stored" else os.path.basename(run_path), args.errors))
    print("\n" + pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
