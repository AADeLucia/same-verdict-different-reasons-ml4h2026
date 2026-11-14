"""Score disagreement_analysis.py (d1) runs against the human/adjudicated FP and FN labels.

Two checks (notes/rq2_disagreement_reanalysis_plan.md, "Validation"):

``fp70``: the 70 blinded FP-adjudication items. Joins each run's ``disagreement_label_derived`` to the
adjudication labels (warranted / supplementary / already_addressed / not_warranted) on (ID, Metric,
Eval Model) and reports how well the two error labels recover the adjudicated errors:
already_addressed → ``already_addressed``, not_warranted (all three are misread questions) →
``misreads_question``. Also prints the cross-tab against the old classifier codes in the key.

``resampled``: the human-corrected old codes (``human_category``) in the resampled annotated files,
mapped to the new labels (FN01 → not_mentioned; FN02 → names_as_missing; FN03/FN05 →
considers_covered; FP01/FP02/FP03 → topic:patient_information / topic:clinical_detail /
topic:seek_care). Raw agreement on mapped rows only: a sanity check, not a reliability figure.

Usage:
    python tools/score_disagreement.py fp70 RUN.jsonl [RUN2.jsonl ...]
    python tools/score_disagreement.py resampled RUN.jsonl
"""
import argparse
import glob
import os

import pandas as pd

Q = "results/reframe/tables_media/qualitative_analysis"
BLIND_DIR = os.path.join(Q, "ml4h_blind")
KEY_COLS = ["ID", "Metric", "Eval Model"]
ERROR_MAP = {"already_addressed": "already_addressed", "not_warranted": "misreads_question"}
OLD_TO_NEW = {
    "FN01": "not_mentioned", "FN02": "names_as_missing", "FN03": "considers_covered", "FN05": "considers_covered",
    "FP01": "topic:patient_information", "FP02": "topic:clinical_detail", "FP03": "topic:seek_care",
}


def load_run(path: str) -> pd.DataFrame:
    run = pd.read_json(path, lines=True)
    parsed = run["model_response_parsed"].map(lambda p: p if isinstance(p, dict) else {})
    run["label"] = run["disagreement_label_derived"]
    omissions = parsed.map(lambda p: [o for o in (p.get("judge_omissions") or []) if isinstance(o, dict)])
    # in_answer summary over the Judge's omissions: "present 2/3" = 2 of 3 omissions present
    run["in_answer"] = omissions.map(
        lambda os_: f"present {sum(o.get('in_answer') == 'present' for o in os_)}/{len(os_)}" if os_ else None
    )
    run["main_in_answer"] = omissions.map(lambda os_: os_[0].get("in_answer") if os_ else None)
    run["topic"] = parsed.map(lambda p: p.get("topic"))
    run["misread_reason"] = parsed.map(lambda p: p.get("misread_reason"))
    run["main_concern"] = omissions.map(lambda os_: os_[0].get("concern") if os_ else None)
    keep = KEY_COLS + ["verdict_type", "label", "in_answer", "main_in_answer", "topic", "misread_reason", "main_concern",
                       "parse_error", "ungrounded_concerns"]
    return run[[c for c in keep if c in run]]


def score_fp70(runs: list) -> None:
    key = pd.read_json(os.path.join(BLIND_DIR, "rq2_blind_validation_key.jsonl"), lines=True)
    ann = pd.read_json(os.path.join(BLIND_DIR, "rq2_blind_validation_annotator1-tp_opus-fp_annotated.jsonl"), lines=True)
    key = key[(key["item_type"] == "fp_adjudication") & ~key["is_retest"].astype(bool)]
    gold = key.merge(ann[["item_id", "human_label", "human_notes"]], on="item_id", validate="1:1")
    for path in runs:
        df = gold.merge(load_run(path), on=KEY_COLS, how="left", validate="1:1")
        print(f"\n=== {path}  (n={len(df)}, missing={df['label'].isna().sum()}, "
              f"parse errors={int(df['parse_error'].fillna(True).sum())})")
        print(pd.crosstab(df["human_label"], df["label"].fillna("NONE"), margins=True).to_string())
        print("\nMain concern in_answer by adjudication:")
        print(pd.crosstab(df["human_label"], df["main_in_answer"].fillna("NONE")).to_string())
        for human, new in ERROR_MAP.items():
            tp = ((df["human_label"] == human) & (df["label"] == new)).sum()
            n_gold = (df["human_label"] == human).sum()
            n_pred = (df["label"] == new).sum()
            print(f"{new:<20} recall {tp}/{n_gold}   precision {tp}/{n_pred}")
        print("\nOld classifier code × new label:")
        print(pd.crosstab(df["classifier_label"], df["label"].fillna("NONE")).to_string())
        print("\nError-label mismatches:")
        for _, r in df.iterrows():
            gold_err = ERROR_MAP.get(r["human_label"])
            pred_err = r["label"] if r["label"] in ERROR_MAP.values() else None
            if gold_err != pred_err:
                print(f"  {r['item_id']}  human={r['human_label']:<17} new={r['label']}  in_answer={r['in_answer']}")
                print(f"      main concern: {str(r['main_concern'])[:160]}")
                print(f"      human note:   {str(r['human_notes'])[:160]}")
                if r["misread_reason"]:
                    print(f"      misread:      {str(r['misread_reason'])[:160]}")


def score_resampled(path: str) -> None:
    files = sorted(glob.glob(os.path.join(Q, "annotated", "rq2_qualitative_validation_*_resampled_annotated.jsonl")))
    gold = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    gold = gold[gold["human_category"].fillna("").str.strip() != ""]
    gold = gold.drop_duplicates(KEY_COLS)[KEY_COLS + ["human_category"]]
    df = load_run(path).merge(gold, on=KEY_COLS, how="inner")
    df["expected"] = df["human_category"].map(OLD_TO_NEW)
    print(f"\n=== {path}  (joined {len(df)} rows)")
    print(pd.crosstab(df["human_category"], df["label"].fillna("NONE"), margins=True).to_string())
    for vt in ("FP", "FN"):
        sub = df[(df["verdict_type"] == vt) & df["expected"].notna()]
        agree = (sub["expected"] == sub["label"]).sum()
        print(f"{vt}: raw agreement on mapped rows {agree}/{len(sub)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["fp70", "resampled"])
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    if args.mode == "fp70":
        score_fp70(args.runs)
    else:
        for path in args.runs:
            score_resampled(path)


if __name__ == "__main__":
    main()
